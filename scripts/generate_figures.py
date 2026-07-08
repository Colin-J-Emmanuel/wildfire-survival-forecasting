"""
Generate the repository figures.

Two kinds:
  * REAL figures, computed directly from our scoring submission
    (submissions/submission_colin.csv). Safe to publish — this is our actual output.
  * One CONCEPTUAL figure (the calibration reliability diagram) that illustrates the
    single most important idea in the project. It is clearly labeled as illustrative;
    regenerate the real version from the walkthrough notebook once the competition
    data is in place.

Usage:  python scripts/generate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SUB = ROOT / "submissions" / "submission_colin.csv"

# --- shared style ---------------------------------------------------------- #
HORIZONS = [12, 24, 48, 72]
COLS = [f"prob_{h}h" for h in HORIZONS]
RED, TEAL, AMBER, SLATE = "#d1495b", "#3c6e71", "#edae49", "#2f3e46"
GRID = "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=13, fontweight="bold", color=SLATE, loc="left", pad=10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
                color="#6b7280", va="bottom")


def fig_mean_by_horizon(df):
    means = df[COLS].mean().values
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(HORIZONS))
    bars = ax.bar(x, means, color=[TEAL, TEAL, AMBER, RED], width=0.62)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.012, f"{m:.2f}",
                ha="center", va="bottom", fontweight="bold", color=SLATE)
    ax.set_xticks(x, [f"{h}h" for h in HORIZONS])
    ax.set_ylim(0, max(means) * 1.25)
    ax.set_ylabel("mean predicted hit probability")
    ax.set_xlabel("forecast horizon")
    _title(ax, "Predicted threat rises with the horizon",
           "Mean P(fire reaches an evacuation zone) across 95 test fires")
    fig.savefig(FIG / "mean_probability_by_horizon.png")
    plt.close(fig)


def fig_trajectories(df):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(HORIZONS))
    order = df[COLS[-1]].values
    cmap = plt.cm.RdYlBu_r
    for _, row in df.iterrows():
        vals = row[COLS].values.astype(float)
        ax.plot(x, vals, color=cmap(row[COLS[-1]]), alpha=0.35, lw=1.1)
    ax.set_xticks(x, [f"{h}h" for h in HORIZONS])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("predicted hit probability")
    ax.set_xlabel("forecast horizon")
    _title(ax, "Every fire's risk is monotone across horizons",
           "One line per fire (12h -> 72h). Enforced guardrail: risk can only rise as a fire nears.")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("72h risk", fontsize=9)
    fig.savefig(FIG / "risk_trajectories.png")
    plt.close(fig)


def fig_risk_composition(df):
    # Share of fires in low / moderate / high risk bands per horizon.
    bands = [("low  (<0.10)", lambda s: s < 0.10, "#cfe1e6"),
             ("moderate", lambda s: (s >= 0.10) & (s < 0.90), AMBER),
             ("high  (>=0.90)", lambda s: s >= 0.90, RED)]
    shares = {name: [f(df[c]).mean() for c in COLS] for name, f, _ in bands}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(HORIZONS))
    bottom = np.zeros(len(HORIZONS))
    for (name, _f, color) in bands:
        vals = np.array(shares[name])
        ax.bar(x, vals, bottom=bottom, color=color, width=0.62, label=name,
               edgecolor="white", linewidth=0.7)
        bottom += vals
    ax.set_xticks(x, [f"{h}h" for h in HORIZONS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of fires")
    ax.set_xlabel("forecast horizon")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16))
    _title(ax, "Fires migrate into the high-risk band over time",
           "Two clear populations: distant fires stay near zero; closing fires resolve toward 1.")
    fig.savefig(FIG / "risk_composition_by_horizon.png")
    plt.close(fig)


def fig_calibration_conceptual():
    # CONCEPTUAL illustration of the core insight — NOT fit on competition data.
    p = np.linspace(0, 1, 60)
    overconfident = np.clip(p ** 1.9, 0, 1)          # predicts high, reality lower
    calibrated = np.clip(p * 0.97 + 0.015, 0, 1)     # near the diagonal
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#9aa5ad", lw=1.2, label="perfect calibration")
    ax.plot(p, overconfident, color=RED, lw=2.4, label="baseline (overconfident)")
    ax.plot(p, calibrated, color=TEAL, lw=2.4, label="after per-horizon calibration")
    ax.fill_between(p, overconfident, p, color=RED, alpha=0.08)
    ax.annotate("overconfidence gap\n(distant fires assigned\nrisk that never materializes)",
                xy=(0.72, 0.72**1.9), xytext=(0.30, 0.80), fontsize=9, color=SLATE,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed hit frequency")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _title(ax, "Why calibration was the biggest win",
           "Conceptual illustration - regenerate real curves from the notebook with competition data")
    ax.text(0.02, -0.14, "Illustrative schematic, not fit on competition data",
            transform=ax.transAxes, fontsize=8, style="italic", color="#9aa5ad")
    fig.savefig(FIG / "calibration_reliability_conceptual.png")
    plt.close(fig)


def main():
    df = pd.read_csv(SUB)
    fig_mean_by_horizon(df)
    fig_trajectories(df)
    fig_risk_composition(df)
    fig_calibration_conceptual()
    fig_social_banner()
    print("Wrote figures to", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  -", p.name)



# --------------------------------------------------------------------------- #
# Social-preview banner (GitHub repo "social preview", 1280x640)
# --------------------------------------------------------------------------- #
def fig_social_banner(rank=477, total=1754):
    import matplotlib.patches as mpatches
    top_pct = round(rank / total * 100)
    W, H = 1280, 640
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(SLATE); ax.set_facecolor(SLATE)
    ax.add_patch(mpatches.Rectangle((0, 0), W, H, color=SLATE, zorder=0))

    # decorative monotone trajectories on the right (echoes risk_trajectories.png)
    rng = np.random.default_rng(7)
    xs = np.array([880, 1000, 1120, 1230])
    for _ in range(26):
        base = rng.uniform(0, 1)
        ys = np.sort(np.clip(base + np.cumsum(rng.uniform(0, 0.35, 4)) - 0.1, 0, 1))
        ys = 90 + ys * 440
        ax.plot(xs, ys, color=plt.cm.RdYlBu_r(ys.max() / 640), alpha=0.22, lw=1.6, zorder=1)

    # fire-gradient accent bar along the bottom
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=[0, W, 0, 10], aspect="auto", cmap="inferno", zorder=2)

    # eyebrow
    ax.text(80, 548, "W I D S   G L O B A L   D A T A T H O N   2 0 2 6",
            color=AMBER, fontsize=13, fontweight="bold", zorder=3)
    # title
    ax.text(78, 470, "Predicting Time-to-Threat", color="white",
            fontsize=44, fontweight="bold", zorder=3)
    ax.text(78, 410, "for Wildfire Evacuation Zones", color="white",
            fontsize=44, fontweight="bold", zorder=3)
    # subtitle
    ax.text(80, 350, "Multi-horizon survival forecasting  ·  12 / 24 / 48 / 72 hours",
            color="#c7d0d4", fontsize=17, zorder=3)

    # metric chips
    chips = [
        ("Score  0.874 → 0.964", RED),
        (f"Top {top_pct}%   ·   {rank} / {total} teams", TEAL),
        ("Break Through Tech AI", "#55606a"),
    ]
    x = 80
    for label, color in chips:
        w = 26 + len(label) * 10.7
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 150), w, 56, boxstyle="round,pad=6,rounding_size=14",
            linewidth=0, facecolor=color, zorder=3))
        ax.text(x + w / 2, 178, label, color="white", fontsize=15,
                fontweight="bold", ha="center", va="center", zorder=4)
        x += w + 20

    fig.savefig(FIG / "social_preview.png", facecolor=SLATE, dpi=100,
                bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
