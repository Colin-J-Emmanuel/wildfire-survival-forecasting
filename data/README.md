# Data

**The competition data is not included in this repository.** It is governed by the WiDS Global
Datathon's *Attribution-NonCommercial-NoDerivatives* license and Kaggle's competition rules, which do
not permit redistribution. You must download it yourself after accepting the rules on Kaggle.

## How to get it

1. Create a (free) Kaggle account and open the competition:
   **WiDS Global Datathon 2026** → *Data* tab.
2. Accept the competition rules.
3. Download the data files and place them in this `data/` directory:

   ```
   data/
   ├── train.csv        # training fires with survival labels (event, time_to_hit_hours)
   ├── test.csv         # fires to predict (no labels)
   └── ...              # any additional provided files
   ```

   Or, with the Kaggle CLI:

   ```bash
   pip install kaggle
   # place your kaggle.json API token in ~/.kaggle/ first
   kaggle competitions download -c WiDSWorldWide_GlobalDathon26 -p data/
   unzip data/*.zip -d data/
   ```

   > Double-check the competition slug (`-c ...`) against the actual Kaggle URL.

## What the data looks like

- **~316 wildfire events** with verified early-stage perimeter observations and confirmed outcomes.
- **Features** are computed strictly from the **first 5 hours** after initial perimeter detection:
  early spread dynamics and the spatial relationship to evacuation zones.
- **Labels** (train only) are **right-censored survival** outcomes:
  - `event` = 1 if the fire came within 5 km of an evacuation-zone centroid within 72h, else 0.
  - `time_to_hit_hours` = the observed hit time, or the censoring time (72) if it never hit.

## Aligning column names

The code references column names in [`src/config.py`](../src/config.py) (e.g. `distance_to_zone_km`,
`spread_rate_kmh`). These reflect the *documented* schema, not necessarily the exact headers in the
released files. After downloading, open `train.csv`, compare the headers, and update the constants in
`src/config.py` (`ID_COL`, `EVENT_COL`, `TIME_COL`, `RAW_FEATURES`) to match. Everything downstream
follows from that one file.
