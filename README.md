# Oslo Air Temperature Inference

Predicts near-surface air temperature (°C) across Oslo from Landsat satellite imagery using a pre-trained XGBoost model. Input predictor stacks are exported from Google Earth Engine and processed locally to produce a single-band GeoTIFF per scene.

---

## How it works

Each Landsat scene is exported from Google Earth Engine as a 9-band GeoTIFF containing the following predictors:

| Band | Description |
|------|-------------|
| `LST` | Land Surface Temperature (Kelvin) |
| `NDVI150` | Normalised Difference Vegetation Index |
| `TPW` | Total Precipitable Water (scene-constant) |
| `Albedo` | Broadband surface albedo |
| `bultFra` | Building fraction |
| `treeFracti` | Tree cover fraction |
| `shade` | Monthly shade fraction (from 1 m LiDAR DEM) |
| `elev30` | Elevation (metres) |
| `aspect30` | Aspect (degrees) |

The inference script applies the model pixel-by-pixel and writes a single-band `Ta_Oslo_YYYY-MM-DD.tif` for each scene.

LST is computed in Google Earth Engine using the open-source module by [Ermida et al. (2020)](https://doi.org/10.3390/rs12091471).

---

## Repository structure

```
oslo-heat-inference/
    oslo_heat_inference.py   ← inference script
    OsloTempModel30m.pkl     ← pre-trained XGBoost model
    README.md
    .gitignore
    stacks/                  ← place input GeoTIFFs here
    predictions/             ← output GeoTIFFs are saved here
```

---

## Requirements

Python 3.9 or later. Install dependencies with pip:

```bash
pip install numpy rasterio xgboost joblib
```

Or with conda:

```bash
conda install numpy rasterio xgboost joblib
```

---

## Usage

**1. Export predictor stacks from Google Earth Engine**

Use the companion GEE script (https://code.earthengine.google.com/84bc748856504b173f73f27c04c53717) to export one 9-band GeoTIFF per Landsat scene to your Google Drive. Download the exported files and place them in the `stacks/` folder.

**2. Run inference**

```bash
python oslo_heat_inference.py
```

Output files are saved to `predictions/` and named `Ta_Oslo_YYYY-MM-DD.tif`. If a scene has already been processed it will be skipped, so the script is safe to re-run.

**3. Adjusting settings**

Open `oslo_heat_inference.py` and edit the `CONFIGURATION` section at the top to change input/output paths or the list of tile codes to skip.

---

## Output

Each output GeoTIFF contains a single band of predicted air temperature in degrees Celsius, in the same CRS and extent as the input stack (EPSG:32632, 30 m resolution).

---

## Citation

If you use this code or model in your work, please cite:

> Gray, A. et al. (in prep). Near-surface air temperature mapping across Oslo using Landsat imagery and machine learning.

LST computation:

> Ermida, S.L., Soares, P., Mantas, V., Göttsche, F.-M., Trigo, I.F., 2020. Google Earth Engine open-source code for Land Surface Temperature estimation from the Landsat series. *Remote Sensing*, 12(9), 1471. https://doi.org/10.3390/rs12091471

---

## Contact

Norwegian Institute for Nature Research (NINA)
