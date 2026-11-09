# Seeing through the clouds: a leakage-controlled, spatially held-out test of Sentinel-1 radar for optical vegetation gaps in Burundi

Does Sentinel-1 radar help fill cloud gaps in Sentinel-2 NDVI, beyond what interpolation, climatology and rainfall already give?
Real clear observations are hidden in gaps of 1, 3 and 6 months and recovered by climatology, linear interpolation, gradient-boosted models with and
without radar, and a radar-only model. *Status: preprint draft, not peer reviewed.* Paper: [`paper/paper.pdf`](paper/paper.pdf).

## Design
7,700 stratified points (7 land-cover classes), monthly 2019 to 2025. Training on 2019 to 2023; testing on 2024 to 2025 at 0.25 degree blocks never seen in training (5 folds, 167,618 targets per gap length). Climatology is computed without the target's own year; confidence intervals use a pixel-cluster bootstrap; uncertainty uses cross-conformal intervals, global and class-conditional.

## Main findings
* Radar helps modestly: RMSE falls by 2.2% (1-month gaps), 3.4% (3 months) and 5.4% (6 months) relative to the same model without radar (all 95% intervals exclude zero), and by 10% to 34% relative to linear interpolation, most of which comes from combining interpolation, climatology and rainfall.
* The gain is class-dependent: about 12% in wetlands at 6 months, 4% in cropland, about zero in built-up land. Radar alone cuts climatology's RMSE by 14%.
* Natural gaps here are short (86% last one month, 2.4% three or more), so the realistic gain is near 2%.
* A global conformal interval has 90% coverage overall but 81% in wetlands and 96% in built-up land; class-conditional calibration gives about 90% everywhere.
* **A check on an unseen 100 m image tile (23 months) reproduced the advantage of the learned models over interpolation (25%) but not the radar gain on its own (0.75%, interval -0.5 to 2.0%).** The paper reports this limitation openly.

## Reproduce
1. `pip install -r requirements.txt`. All data used are in `data/` (re-export with the two `.js` scripts in the Earth Engine Code Editor if desired).
2. From the repository root: `python build.py` (merges the CSVs into `panel.pkl`), `python exp.py 1`, `python exp.py 3 6` (about 5 minutes), `python analyze.py`, `python make_figs.py`.
3. Image-tile check: `python tile_check.py`, then `python fig_tile.py`.
4. `cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`.

Random seeds are fixed. Data terms: `DATA.md`. Code MIT, paper CC BY 4.0, cite via `CITATION.cff`. Contact: Kana Amédée, kanaamedee00@gmail.com.
