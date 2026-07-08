# Figures

These are generated, not hand-drawn. Rebuild them any time with:

```bash
python scripts/generate_figures.py
```

| File | Source | What it shows |
|---|---|---|
| `mean_probability_by_horizon.png` | **real** — our submission | Mean predicted hit probability rising across 12/24/48/72h. |
| `risk_trajectories.png` | **real** — our submission | One line per fire; risk is monotone across horizons; two fire populations. |
| `risk_composition_by_horizon.png` | **real** — our submission | Share of fires in low/moderate/high risk bands, and how they migrate over time. |
| `calibration_reliability_conceptual.png` | **conceptual** | Illustrates the distant-fire overconfidence fix. *Not fit on competition data* — regenerate the real reliability diagram from the walkthrough notebook once the Kaggle data is in place. |
| `social_preview.png` | branding | 1280×640 hero/social-preview banner (title, result, leaderboard placement). Set it in GitHub → Settings → General → Social preview. |

The "real" figures are computed directly from `submissions/submission_colin.csv`, so they're safe to
publish and accurately represent our output. The conceptual one is clearly labeled as illustrative in the
image itself.
