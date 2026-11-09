# Data provenance and licences

All files in `data/` were exported by the author from Google Earth Engine (scripts `burundi_sar_optical_export.js` and `gapfill_images_export.js`):

* `data/gapfill_*.csv`: 7,700 stratified points across Burundi, monthly 2019 to 2025: Sentinel-2 NDVI (median of clear observations), Sentinel-1 VV and VH backscatter, CHIRPS rainfall, SRTM elevation and slope, ESA WorldCover 2021 class (used only to stratify the sample). Values are integers scaled as described in the export script (NDVI x 1000, dB x 100, -32768 = missing).
* `data/tile/`: a 100 m monthly tile of Bujumbura and the Imbo plain (84 months x NDVI, VV, VH), static layers, a true-colour composite for July and November 2024, and monthly rainfall.

Sources and terms: Copernicus Sentinel-1 and Sentinel-2 (contains modified Copernicus Sentinel data; free and open under the Copernicus terms), CHIRPS (Funk et al., 2015), SRTM (NASA), ESA WorldCover 2021 v200 (Zanaga et al., 2022, CC BY 4.0, doi:10.5281/zenodo.7254221). Check each dataset's Earth Engine catalogue page for current terms before reuse.
