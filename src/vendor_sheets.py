# -*- coding: utf-8 -*-
"""
Vendor call sheets: one formatted Excel workbook per vendor + a management summary.

Input : data/clients_to_contact.csv
Output: data/fiches_vendeurs/fiche_<code>_<nom>.xlsx   (one per vendor — a vendor only
        receives their own clients, per the permission principle in docs/background/project.md §12)
        data/synthese_direction.xlsx                   (per-vendor overview for management)

Each sheet has an "Action vendeur" dropdown + free comment column: this is the manual
version of the feedback loop (docs/background/plan.md Phase 2) — usable before any application exists.
"""
import io
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "fiches_vendeurs"
OUT.mkdir(exist_ok=True)

STATUS_FR = {"INACTIVE_2026": "Inactif 2026", "DECLINING": "En baisse", "ACTIVE": "Actif"}
STATUS_FILL = {
    "INACTIVE_2026": PatternFill("solid", start_color="F4CCCC"),  # red
    "DECLINING": PatternFill("solid", start_color="FCE5CD"),      # orange
    "ACTIVE": PatternFill("solid", start_color="D9EAD3"),         # green
}
HEADER_FILL = PatternFill("solid", start_color="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)

HEADERS = ["N°", "Client", "Statut", "Priorité (TND)", "Dernière activité", "Résumé",
           "Article recommandé 1", "Raison 1", "Article recommandé 2", "Raison 2",
           "Article recommandé 3", "Raison 3", "À proposer aussi",
           "Action vendeur", "Commentaire"]
WIDTHS = [5, 32, 13, 14, 13, 38, 45, 42, 45, 42, 45, 42, 45, 16, 28]
ACTIONS = '"Contacté,Devis envoyé,Commande,À rappeler,Refus,Client fermé,Injoignable"'


def _with_proba(row, i: int) -> str:
    """Reason text prefixed with the model's repurchase probability.

    Marked as an estimate because it is known to run optimistic: an isotonic correction
    was measured and rejected (reports/model_catboost.md), so the number is shown raw and
    must never appear without that qualifier.
    """
    reason = row.get(f"reason_{i}")
    reason = "" if pd.isna(reason) else str(reason)
    proba = row.get(f"proba_{i}")
    if pd.isna(proba):
        return reason
    return f"{proba:.0f} % de chance qu'il le reprenne (estimation optimiste). {reason}".strip()


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")
    return s[:40] or "vendeur"


def write_vendor_file(code: str, name: str, rows: pd.DataFrame) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "À contacter"

    n = len(rows)
    tnd = rows["priority_score"].sum()

    ws.merge_cells("A1:H1")
    ws["A1"] = f"Clients à contacter — {name or code}"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A2:H2")
    ws["A2"] = (f"Généré le {date.today():%d/%m/%Y} — {n} clients — "
                f"{tnd:,.0f} TND en jeu — trié par priorité")
    ws["A2"].font = Font(italic=True, color="666666")

    hr = 4
    for j, (h, w) in enumerate(zip(HEADERS, WIDTHS), start=1):
        cell = ws.cell(row=hr, column=j, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = f"A{hr + 1}"

    dv = DataValidation(type="list", formula1=ACTIONS, allow_blank=True)
    ws.add_data_validation(dv)

    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        row = hr + i
        values = [
            i, r["client_name"], STATUS_FR.get(r["status"], r["status"]),
            round(r["priority_score"]), r["last_activity"], r["resume"],
            # The probability is prefixed onto the reason so the Excel sheet keeps the
            # same information as the app without needing three extra columns.
            r.get("article_txt_1", ""), _with_proba(r, 1),
            r.get("article_txt_2", ""), _with_proba(r, 2),
            r.get("article_txt_3", ""), _with_proba(r, 3),
            r.get("suggestion_complementaire", ""), "", "",
        ]
        for j, v in enumerate(values, start=1):
            cell = ws.cell(row=row, column=j, value=None if pd.isna(v) else v)
            cell.border = THIN
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=row, column=3).fill = STATUS_FILL.get(r["status"], PatternFill())
        ws.cell(row=row, column=4).number_format = "#,##0"
        dv.add(ws.cell(row=row, column=14))

    path = OUT / f"fiche_{code}_{safe_name(name)}.xlsx"
    wb.save(path)
    return path


def write_summary(q: pd.DataFrame) -> Path:
    s = (
        q.groupby(["vendor_code", "vendor_name"], dropna=False)
        .agg(clients=("client_id", "size"),
             inactifs=("status", lambda x: (x == "INACTIVE_2026").sum()),
             en_baisse=("status", lambda x: (x == "DECLINING").sum()),
             actifs=("status", lambda x: (x == "ACTIVE").sum()),
             tnd_en_jeu=("priority_score", "sum"))
        .reset_index()
        .sort_values("tnd_en_jeu", ascending=False)
    )
    wb = Workbook()
    ws = wb.active
    ws.title = "Synthèse"
    ws.merge_cells("A1:G1")
    ws["A1"] = f"Synthèse par vendeur — {date.today():%d/%m/%Y}"
    ws["A1"].font = Font(bold=True, size=14)

    headers = ["Code", "Vendeur", "Clients à contacter", "Inactifs 2026",
               "En baisse", "Actifs", "TND en jeu"]
    widths = [8, 30, 18, 14, 12, 10, 14]
    for j, (h, w) in enumerate(zip(headers, widths), start=1):
        c = ws.cell(row=3, column=j, value=h)
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = "A4"

    for i, (_, r) in enumerate(s.iterrows(), start=4):
        vals = [r["vendor_code"], r["vendor_name"], r["clients"], r["inactifs"],
                r["en_baisse"], r["actifs"], round(r["tnd_en_jeu"])]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=i, column=j, value=None if pd.isna(v) else v)
            c.border = THIN
        ws.cell(row=i, column=7).number_format = "#,##0"

    total_row = len(s) + 4
    ws.cell(row=total_row, column=2, value="TOTAL").font = Font(bold=True)
    ws.cell(row=total_row, column=3, value=int(s["clients"].sum())).font = Font(bold=True)
    c = ws.cell(row=total_row, column=7, value=round(float(s["tnd_en_jeu"].sum())))
    c.font = Font(bold=True)
    c.number_format = "#,##0"

    path = DATA / "synthese_direction.xlsx"
    wb.save(path)
    return path


def main():
    q = pd.read_csv(DATA / "clients_to_contact.csv",
                    dtype={"vendor_code": str, "client_id": str}, encoding="utf-8-sig")
    q = q.sort_values(["vendor_code", "priority_score"], ascending=[True, False])

    files = []
    for (code, name), rows in q.groupby(["vendor_code", "vendor_name"], dropna=False):
        files.append(write_vendor_file(code, "" if pd.isna(name) else name, rows))
    summary = write_summary(q)

    print(f"vendor files : {len(files)}  ->  {OUT}")
    print(f"summary      : {summary}")
    print(f"clients total: {len(q):,}  |  TND en jeu: {q.priority_score.sum():,.0f}")


if __name__ == "__main__":
    main()
