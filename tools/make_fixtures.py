# -*- coding: utf-8 -*-
"""Build fixtures/ — an anonymised, reduced copy of data/ that is safe to commit.

Why this exists
---------------
A frontend developer needs the *shape* of the API responses, never the contents:
data/ holds real client names and real revenue, which is why .gitignore excludes it.
This script produces a dataset with the same columns, the same dtypes and the same
response shapes, with every identity replaced.

    py tools/make_fixtures.py
    DATA_ROOT=fixtures py -m uvicorn api.main:app --host 127.0.0.1 --port 8000

Run it yourself; the developer never does — they receive the committed output.

What is anonymised, and what is not
-----------------------------------
Replaced : vendor / client / article identifiers, vendor and client names, article
           brands (enseigne) and designations.
Kept     : the money, the months, the model probabilities. Once an amount can no longer
           be attributed to a named client it is no longer confidential, and keeping it
           preserves internal consistency — the reason texts quote figures that must
           still match the rows they describe.
Copied   : reports/rolling_metrics.json, which contains aggregate metrics only and is
           already committed to the repository.

The output is guarded by tests/test_no_data_leak.py, which fails if any real name
survives into fixtures/.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DATA = ROOT / "data"
SRC_REPORTS = ROOT / "reports"
OUT = ROOT / "fixtures"
OUT_DATA = OUT / "data"
OUT_REPORTS = OUT / "reports"

# Small enough to start instantly, large enough that decile_table's qcut(10) and the
# calibration bins still have rows to work with. Capping clients per vendor matters more
# than the vendor count: one large vendor alone pulls in hundreds of clients and tens of
# thousands of articles, which makes the fixture heavier than the real extract is useful.
N_VENDORS = 10
MAX_CLIENTS_PER_VENDOR = 8

# Synthetic designations mirror the real structure — part type, reference, vehicle —
# because the frontend has to lay out strings of that shape.
#
# The part types must not be real designations in their own right. "POMPE A EAU" and
# "RADIATEUR EAU" were, once the catalogue grew from 24,683 to 52,115 articles, and
# tests/test_no_data_leak.py duly failed. The old comment claimed collision was
# impossible because the VEHICLE names were invented; that was only ever true by
# luck. assert_no_real_designation() below now enforces it at generation time
# instead of asserting it in prose.
PART_TYPES = [
    "PLAQUETTE FREIN AV", "FILTRE A HUILE", "COURROIE DISTRIBUTION",
    "ROULEMENT ROUE AR", "AMORTISSEUR AV", "BOUGIE ALLUMAGE", "DISQUE FREIN AV",
    "POMPE CIRCUIT ZEPHYR", "KIT EMBRAYAGE", "FILTRE A AIR", "RADIATEUR CIRCUIT ZEPHYR",
    "ALTERNATEUR", "HUILE MOTEUR 5W40 5L", "LIQUIDE REFROIDISSEMENT 1L",
    "CYLINDRE DE ROUE", "JEU DE MACHOIRES", "TRIANGLE SUSPENSION",
    "SONDE TEMPERATURE", "POMPE A CARBURANT", "MAITRE CYLINDRE",
]
VEHICLES = ["MODELE ALPHA", "MODELE BETA 1.5", "TYPE GAMMA-200", "SERIE DELTA",
            "GAMME EPSILON 4", "MODELE ZETA DCI", "TYPE ETA 2.0"]


def fail(msg: str) -> None:
    sys.exit(f"[fixtures] {msg}")


def assert_no_real_designation(part_types, real_designations) -> None:
    """Fail generation if a synthetic part type contains a real designation.

    The alternative is silently shipping catalogue text into a tree that
    tests/test_no_data_leak.py scans - i.e. finding out at test time, in a
    failure whose cause looks nothing like its symptom.
    """
    # 13 mirrors tests/test_no_data_leak.py::MIN_DESIGNATION exactly. A stricter
    # bar here would block generic words like "ROULEMENT" that the leak test
    # never flags, i.e. enforce a standard nothing actually requires.
    real = {d.upper() for d in real_designations
            if isinstance(d, str) and len(d) >= 13}
    for part in part_types:
        hit = next((d for d in real if d in part.upper()), None)
        if hit:
            fail(f"part type {part!r} contains the real designation {hit!r} - "
                 f"rename it in PART_TYPES.")


def main() -> None:
    if not (SRC_DATA / "clients_to_contact.csv").exists():
        fail("data/clients_to_contact.csv missing — run the pipeline first.")

    src_articles = pd.read_csv(SRC_DATA / "articles.csv", dtype=str,
                               encoding="utf-8-sig")
    assert_no_real_designation(PART_TYPES, src_articles["designation"].dropna())

    ids = {"vendor_code": str, "client_id": str, "article_id": str}
    queue = pd.read_csv(SRC_DATA / "clients_to_contact.csv", dtype=ids,
                        encoding="utf-8-sig")
    fact = pd.read_parquet(SRC_DATA / "ca_monthly.parquet")
    # The LIVE scores, matching what the API serves. pair_scores_rolling.csv is the
    # backtest — it carries the answer key in `label` and a window that has already
    # elapsed, so shipping it as a fixture invites a frontend to build against a
    # shape the API no longer returns.
    scores = {"live": pd.read_csv(SRC_DATA / "pair_scores_live.csv",
                                  dtype=ids, encoding="utf-8-sig")}

    # ---- pick a spread of vendors, not just the biggest ones -------------------
    by_size = queue["vendor_code"].value_counts()
    step = max(1, len(by_size) // N_VENDORS)
    vendors = sorted(by_size.index[::step][:N_VENDORS])
    queue = queue[queue["vendor_code"].isin(vendors)].copy()
    # Keep each vendor's highest-priority clients: the queue stays representative of what
    # a vendor actually sees first, rather than an arbitrary slice of the tail.
    queue = (queue.sort_values(["vendor_code", "priority_score"], ascending=[True, False])
                  .groupby("vendor_code", group_keys=False)
                  .head(MAX_CLIENTS_PER_VENDOR)
                  .copy())
    clients = sorted(queue["client_id"].unique())

    fact = fact[fact["client_id"].isin(clients)].copy()
    for s in scores:
        scores[s] = scores[s][scores[s]["client_id"].isin(clients)].copy()

    # ---- identifier maps -------------------------------------------------------
    def first_token(value):
        """article_txt / suggestion are '<article_id> <enseigne> — <designation>'."""
        return value.split(" ", 1)[0] if isinstance(value, str) and value.strip() else None

    referenced = set()
    for col in ["article_txt_1", "article_txt_2", "article_txt_3",
                "suggestion_complementaire"]:
        referenced |= {t for t in queue[col].map(first_token) if t}
    referenced |= set(fact["article_id"])
    for s in scores.values():
        referenced |= set(s["article_id"])

    vmap = {v: f"V{i:02d}" for i, v in enumerate(vendors, 1)}
    cmap = {c: f"C{i:03d}" for i, c in enumerate(clients, 1)}
    amap = {a: f"A{i:04d}" for i, a in enumerate(sorted(referenced), 1)}

    def article_row(old_id: str, new_id: str) -> dict:
        n = int(new_id[1:])
        return {
            "article_id": new_id,
            "enseigne": f"MRQ{n % 12 + 1:02d}",
            "designation": (f"{PART_TYPES[n % len(PART_TYPES)]}"
                            f"-{1000000 + n * 7:07d}-{VEHICLES[n % len(VEHICLES)]}"),
        }

    articles = pd.DataFrame([article_row(o, n) for o, n in sorted(amap.items(),
                                                                 key=lambda kv: kv[1])])
    alabel = {r.article_id: f"{r.article_id} {r.enseigne} — {r.designation}"
              for r in articles.itertuples()}

    # ---- rewrite the queue -----------------------------------------------------
    queue["vendor_code"] = queue["vendor_code"].map(vmap)
    queue["client_id"] = queue["client_id"].map(cmap)
    queue["vendor_name"] = queue["vendor_code"].str.replace("V", "Vendeur ", regex=False)
    queue["client_name"] = queue["client_id"].str.replace("C", "Client ", regex=False)
    for col in ["article_txt_1", "article_txt_2", "article_txt_3",
                "suggestion_complementaire"]:
        queue[col] = queue[col].map(
            lambda v: alabel.get(amap.get(first_token(v), ""), None))

    fact["vendor_code"] = fact["vendor_code"].map(vmap)
    fact["client_id"] = fact["client_id"].map(cmap)
    fact["article_id"] = fact["article_id"].map(amap)
    # A client may be served by a vendor outside the sample; those rows have no code.
    fact = fact.dropna(subset=["vendor_code", "client_id", "article_id"])

    for s, df in scores.items():
        df["vendor_code"] = df["vendor_code"].map(vmap)
        df["client_id"] = df["client_id"].map(cmap)
        df["article_id"] = df["article_id"].map(amap)
        scores[s] = df.dropna(subset=["vendor_code", "client_id", "article_id"])

    # ---- write -----------------------------------------------------------------
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_DATA.mkdir(parents=True)
    OUT_REPORTS.mkdir(parents=True)

    enc = "utf-8-sig"
    queue.to_csv(OUT_DATA / "clients_to_contact.csv", index=False, encoding=enc)
    fact.to_parquet(OUT_DATA / "ca_monthly.parquet", index=False)
    articles.to_csv(OUT_DATA / "articles.csv", index=False, encoding=enc)
    pd.DataFrame({"vendor_code": list(vmap.values()),
                  "vendor_name": [f"Vendeur {v[1:]}" for v in vmap.values()]}
                 ).to_csv(OUT_DATA / "vendors.csv", index=False, encoding=enc)
    pd.DataFrame({"client_id": list(cmap.values()),
                  "client_name": [f"Client {c[1:]}" for c in cmap.values()]}
                 ).to_csv(OUT_DATA / "clients.csv", index=False, encoding=enc)
    for s, df in scores.items():
        df.to_csv(OUT_DATA / f"pair_scores_{s}.csv", index=False, encoding=enc)

    # The serving-window record travels with the scores. n_pairs is rewritten to the
    # fixture's own count: copied verbatim it would claim the full 518k and defeat the
    # provenance check it exists for.
    window = SRC_REPORTS / "live_window.json"
    if window.exists():
        record = json.loads(window.read_text(encoding="utf-8"))
        record["n_pairs"] = int(len(scores["live"]))
        (OUT_REPORTS / "live_window.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
    else:
        print("[fixtures] warning: reports/live_window.json missing — "
              "run src/score_live.py; the API will report an unknown window.")

    metrics = SRC_REPORTS / "rolling_metrics.json"
    if metrics.exists():
        json.loads(metrics.read_text(encoding="utf-8"))     # fail early if malformed
        shutil.copy(metrics, OUT_REPORTS / "rolling_metrics.json")
    else:
        print("[fixtures] warning: reports/rolling_metrics.json missing — "
              "the validation page will error on this set.")

    print(f"[fixtures] vendors {len(vmap)} | clients {len(cmap)} | "
          f"articles {len(amap):,}")
    print(f"[fixtures] fact rows {len(fact):,} | queue rows {len(queue):,} | "
          f"pair_scores live {len(scores['live']):,}")
    # No label mix to report: a live window has no observed outcome. Mean p stands in
    # as the sanity check that the scores survived anonymisation as numbers.
    print(f"[fixtures] mean p: {scores['live']['p'].mean():.4f}")
    print(f"[fixtures] written to {OUT}")


if __name__ == "__main__":
    main()
