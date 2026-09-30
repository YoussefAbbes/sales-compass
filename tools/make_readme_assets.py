"""Render the results chart used in the README from the committed metrics files.

    python tools/make_readme_assets.py        ->  docs/assets/results.png

Numbers are read from reports/*.json, never typed in, so the picture cannot drift
from the model cards.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rolling = json.loads((ROOT / "reports/rolling_metrics.json").read_text(encoding="utf-8"))
prospect = json.loads((ROOT / "reports/prospect_metrics.json").read_text(encoding="utf-8"))

GREEN, ORANGE, SAND, INK, MUTED = "#0f5c4a", "#d64b1f", "#d9d2c3", "#1b1a17", "#6b665b"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, facecolor="#fbf9f4")
panels = [
    ("Repurchase — precision in the top 30",
     [("Sort by revenue\n(what vendors do)", rolling["precision_top30_revenue"], SAND),
      ("Model, as served\n(recency-damped)", rolling["precision_top30_model_damped"], GREEN),
      ("Model, raw score", rolling["precision_top30_model"], GREEN)]),
    ("Article adoption — precision in the top 5",
     [("Random", prospect["rankers"]["random"]["precision_at_k"], SAND),
      ("Client size", prospect["rankers"]["client size"]["precision_at_k"], SAND),
      ("Basket heuristic", prospect["rankers"]["heuristic"]["precision_at_k"], SAND),
      ("CatBoost model", prospect["rankers"]["model"]["precision_at_k"], ORANGE)]),
]
for ax, (title, bars) in zip(axes, panels):
    labels, values, colors = zip(*bars)
    y = range(len(bars))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=12)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.02, yi, f"{v:.1%}", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(SAND)
    ax.set_facecolor("#fbf9f4")
    ax.tick_params(axis="y", length=0)

fig.text(0.01, 0.01,
         f"Held-out windows · repurchase AUC {rolling['auc']['test']:.3f} · adoption AUC {prospect['auc']:.3f} · "
         "scores are rankings, not calibrated probabilities",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=4)
out = ROOT / "docs/assets/results.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print("wrote", out.relative_to(ROOT))
