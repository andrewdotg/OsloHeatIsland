"""
oslo_heat_inference.py
======================
Runs air temperature inference on Landsat predictor stacks
exported from Google Earth Engine, using a pre-trained
XGBoost model.

For each input GeoTIFF this script produces a single-band
output GeoTIFF containing predicted air temperature (°C).

BEFORE YOU START DOWNLOAD INFERENCE STACKS USING THIS GOOGLE EARTH ENGINE SCRIPT
DOWNLOAD THE STACKS FROM GOOGLE DRIVE AND SAVE THEM IN THE FOLDER CALLED "stacks"

https://code.earthengine.google.com/84bc748856504b173f73f27c04c53717


RUNNING THE PYTHO SCRIPT — install dependencies
----------------------------------------
Open a terminal and run:

    pip install numpy rasterio xgboost joblib

If you are using a virtual environment (recommended):

    python -m venv oslo_env
    source oslo_env/bin/activate      # Mac / Linux
    oslo_env\Scripts\activate         # Windows
    pip install numpy rasterio xgboost joblib

Note: xgboost must be installed even though inference is run
through the XGBoost model's standard predict() method, because
joblib needs it to deserialise the saved model file.

Python 3.9 or later is recommended.

USAGE
-----
1. Edit the paths in the CONFIGURATION section below.
2. Run from a terminal:

       python oslo_heat_inference.py

Results are saved to OUTPUT_FOLDER. Output files are named
Ta_Oslo_YYYY-MM-DD.tif. Already-processed files are skipped
automatically, so the script is safe to re-run if interrupted.
"""

import os
import re
import sys

# =============================================================
# CONFIGURATION — edit these before running
# =============================================================

# The script assumes the following layout:
#
#   oslo_heat_inference.py        <- this script
#   OsloTempModel30m.pkl          <- the model file
#   stacks/                       <- folder of input GeoTIFFs
#       oslo_stack_L8_2023-07-06.tif
#       oslo_stack_L9_2023-07-16.tif
#       ...
#   predictions/                  <- output folder (auto-created)
#
# If your files are arranged this way, you do not need to
# change anything below. Otherwise, update the paths.

# Resolves paths relative to the location of this script,
# so the script works regardless of where you run it from.
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER  = os.path.join(SCRIPT_DIR, "stacks")
MODEL_PATH    = os.path.join(SCRIPT_DIR, "OsloTempModel30m.pkl")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "predictions")

# Tile codes to skip — useful for excluding tiles that cover
# areas outside the study boundary (e.g. sea or border tiles).
# Add or remove WRS path/row strings as needed.
SKIP_CODES = ("196019", "196018")

# =============================================================
# FEATURE NAMES
# Must match the band order in your exported GeoTIFFs exactly,
# and must match the feature order used during model training.
# =============================================================

FEATURE_NAMES = [
    "LST",
    "NDVI150",
    "TPW",
    "Albedo",
    "bultFra",
    "treeFracti",
    "shade",
    "elev30",
    "aspect30",
]

# =============================================================
# IMPORTS
# =============================================================

import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit(
        "\nERROR: rasterio is not installed.\n"
        "Fix:   pip install rasterio\n"
    )

try:
    import joblib
except ImportError:
    sys.exit(
        "\nERROR: joblib is not installed.\n"
        "Fix:   pip install joblib\n"
    )

try:
    import xgboost  # noqa -- required to deserialise the saved XGBoost model
except ImportError:
    sys.exit(
        "\nERROR: xgboost is not installed.\n"
        "Fix:   pip install xgboost\n"
    )

# =============================================================
# LOAD MODEL
# =============================================================

if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"\nERROR: Model file not found:\n  {MODEL_PATH}\n"
        "Check that MODEL_PATH is set correctly in the CONFIGURATION section.\n"
    )

print(f"\nLoading model from:\n  {MODEL_PATH}")
rf = joblib.load(MODEL_PATH)
print(f"  Model loaded OK -- type: {type(rf).__name__}")

# =============================================================
# SET UP OUTPUT FOLDER
# =============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =============================================================
# MAIN LOOP -- process each GeoTIFF
# =============================================================

tif_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")])

if not tif_files:
    sys.exit(
        f"\nERROR: No .tif files found in:\n  {INPUT_FOLDER}\n"
        "Check that INPUT_FOLDER is set correctly.\n"
    )

print(f"\nFound {len(tif_files)} GeoTIFF(s) in input folder.")
print(f"Outputs will be saved to:\n  {OUTPUT_FOLDER}\n")

for fname in tif_files:

    # --- Skip unwanted tiles ---
    if any(code in fname for code in SKIP_CODES):
        print(f"  Skipping tile (excluded code): {fname}")
        continue

    in_path = os.path.join(INPUT_FOLDER, fname)

    # Extract date from filename (expects YYYY-MM-DD somewhere in the name)
    # and use it to build the output filename.
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    date_str   = date_match.group(1) if date_match else fname.replace(".tif", "")
    out_path   = os.path.join(OUTPUT_FOLDER, f"Ta_Oslo_{date_str}.tif")

    # --- Resume logic: skip already-processed files ---
    if os.path.exists(out_path):
        print(f"  Already processed, skipping: {fname}")
        continue

    print(f"  Processing: {fname}")

    # --- Read raster ---
    with rasterio.open(in_path) as src:
        data    = src.read()           # shape: (n_bands, n_rows, n_cols)
        profile = src.profile
        n_bands, n_rows, n_cols = data.shape

    # Validate band count
    if n_bands != len(FEATURE_NAMES):
        print(
            f"    WARNING: Expected {len(FEATURE_NAMES)} bands, "
            f"found {n_bands} -- skipping {fname}.\n"
            f"    Check that FEATURE_NAMES matches your GeoTIFF band order."
        )
        continue

    # --- Flatten to pixel array ---
    # data is (bands, rows, cols) -> transpose to (pixels, bands)
    X       = data.reshape(n_bands, -1).T
    mask    = np.any(np.isnan(X), axis=1)  # True where any band is nodata
    X_valid = X[~mask]

    if X_valid.shape[0] == 0:
        print(f"    No valid pixels found -- skipping {fname}.")
        continue

    print(f"    Valid pixels: {X_valid.shape[0]:,} of {n_rows * n_cols:,}")

    # --- Run inference ---
    print("    Running inference...")
    predictions = rf.predict(X_valid)

    # --- Rebuild full-size raster (NaN where masked) ---
    full_pred = np.full(n_rows * n_cols, np.nan, dtype="float32")
    full_pred[~mask] = predictions.astype("float32")

    # --- Write single-band output GeoTIFF ---
    profile.update(count=1, dtype="float32", nodata=np.nan)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(full_pred.reshape(1, n_rows, n_cols))
        dst.update_tags(1, name="predicted_temp_C")

    print(f"    Saved: {os.path.basename(out_path)}")

# =============================================================
# SUMMARY
# =============================================================

print("\nAll files processed.")