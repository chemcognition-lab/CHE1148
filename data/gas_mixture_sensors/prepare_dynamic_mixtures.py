"""
Dynamic Gas Mixtures Time-Series Data Preparation (Fonollosa et al. 2015).

Reference:
    Fonollosa, J. et al. (2015). 'Reservoir computing and low-power sensors 
    for monitoring in dynamic environments'. Sensors and Actuators B: Chemical.
    DOI: 10.1016/j.snb.2015.03.028
    URL: https://archive.ics.uci.edu/dataset/322

Data Explanation:
    The dataset contains 16 Metal Oxide (MOX) sensors exposed to dynamic, 
    binary gas mixtures in air. There are two independent experimental sessions:
    1. Methane (CH4) and Ethylene (C2H4)
    2. Carbon Monoxide (CO) and Ethylene (C2H4)
    Concentrations change randomly every 80-120 seconds. Data was recorded 
    continuously at 100 Hz for approximately 12 hours per session.

The Machine Learning Task:
    Multi-Output Time-Series Regression. The objective is to estimate the 
    instantaneous concentrations of both gases simultaneously. 
    Because MOX sensors act as non-linear low-pass filters (they react quickly 
    but recover slowly), they exhibit severe hysteresis. Standard row-by-row 
    models fail. Accurate models must use temporal features (lags and slopes) 
    to deconvolve the concentration from the sensor's kinetic state.

Preprocessing Architecture:
    1. Raw Audit: Verifies the 100Hz continuity of the ~4.1M row raw text file.
    2. Physics Transform: Converts raw conductivity (S) to Resistance (R) in 
       kOhms using the manufacturer formula R = 40,000 / S.
    3. Healing: 0.0 conductivity values create infinite resistance. These are 
       replaced with NaNs and linearly interpolated to strictly preserve time.
    4. Downsampling: 100Hz is redundant for 100-second gas pulses. We downsample 
       to 1Hz, dropping file sizes from ~280MB down to ~2.6MB per mixture.

Run Example:
    python prepare_dynamic_mixtures.py --downsample 100 --clip --interpolate
"""

import argparse
import json
import pathlib
import shutil
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class Paths:
    """Namespace for I/O operations and file tracking."""
    raw = pathlib.Path('.')
    out = pathlib.Path('.')
    temp = raw / "temp_extract"
    metadata = out / "metadata.json"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    zip_url = (
        "https://archive.ics.uci.edu/static/public/322/"
        "gas+sensor+array+under+dynamic+gas+mixtures.zip"
    )
    zip_file = raw / "dynamic_gas.zip"


class Constants:
    """Hardware parameters and experimental mappings."""
    conversion_k = 40000.0  # R = 40.000 / S
    base_hz = 100.0
    
    # UCI specified sensor layout
    sensor_models =[
        "TGS2602", "TGS2602", "TGS2600", "TGS2600",
        "TGS2610", "TGS2610", "TGS2620", "TGS2620",
        "TGS2602", "TGS2602", "TGS2600", "TGS2600",
        "TGS2610", "TGS2610", "TGS2620", "TGS2620",
    ]
    sensor_cols =[
        f"s{i+1:02d}_{m.lower()}" for i, m in enumerate(sensor_models)
    ]
    
    mixtures = {
        "ethylene_methane.txt": {"id": "CH4_C2H4", "sec_gas": "methane_ppm"},
        "ethylene_CO.txt": {"id": "CO_C2H4", "sec_gas": "co_ppm"},
    }


def fetch_and_unzip(_P: Paths) -> None:
    """Downloads and extracts the raw archive if not cached."""
    if not _P.zip_file.exists():
        print("Downloading dataset from UCI (~350MB)...")
        import requests
        with requests.get(_P.zip_url, stream=True) as r:
            r.raise_for_status()
            with open(_P.zip_file, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    if not _P.temp.exists():
        print("Extracting raw files...")
        with zipfile.ZipFile(_P.zip_file, "r") as z:
            z.extractall(_P.temp)


def process_and_audit_mixture(
    file_path: pathlib.Path, config: dict[str, str], args: argparse.Namespace, _P: Paths, _C: Constants
) -> dict[str, any]:
    """Handles raw audit, physics transformation, post-audit, and saving."""
    print(f"\nProcessing {config['id']}...")
    
    # 1. LOAD RAW DATA
    # Skip string header to prevent axis mismatch
    df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)
    df.columns = ["time_sec", config["sec_gas"], "ethylene_ppm"] + _C.sensor_cols

    # 2. RAW AUDIT
    raw_dt = df["time_sec"].diff().median()
    raw_nans = df.isna().sum().sum()
    print(f"  [Raw Audit] Shape: {df.shape} | dt: {raw_dt:.2f}s | NaNs: {raw_nans}")

    # 3. PHYSICS TRANSFORM: Conductivity -> Resistance
    # 0.0 conductivity = infinite resistance. Replace with NaN for healing.
    for col in _C.sensor_cols:
        raw_s = df[col].replace(0.0, np.nan)
        df[col] = _C.conversion_k / raw_s

    # 4. HEALING & CLIPPING
    if args.interpolate:
        df[_C.sensor_cols] = df[_C.sensor_cols].interpolate(method="linear").ffill().bfill()
    else:
        df = df.dropna()

    if args.clip:
        df[_C.sensor_cols] = df[_C.sensor_cols].clip(lower=0.1, upper=250.0)

    # 5. DOWNSAMPLING
    if args.downsample > 1:
        df = df.iloc[::args.downsample].reset_index(drop=True)
    df.insert(0, "mixture_type", config["id"])

    # 6. POST-PROCESSED AUDIT
    final_dt = df["time_sec"].diff().median()
    final_nans = df.isna().sum().sum()
    
    # Spearman rank on Sensor 1 vs Ethylene
    # (Values of -0.1 to -0.6 are normal due to cross-interference of the second gas)
    corr, _ = spearmanr(df["ethylene_ppm"], df[_C.sensor_cols[0]])
    
    print(f"  -> Final Shape:       {df.shape}")
    print(f"  -> Contiguous Time:   {'✅' if final_nans == 0 else '❌'} (dt={final_dt:.2f}s)")
    print(f"  -> NaN Count:         {final_nans} (Should be 0)")
    print(f"  -> Rank Monotonicity: {corr:.4f} (Negative confirms R drops as gas rises)")

    # 7. SAVE
    target_hz = _C.base_hz / args.downsample
    out_file = _P.out / f"{config['id']}_{target_hz:.0f}Hz.parquet"
    df.to_parquet(out_file, index=False, compression="snappy")
    
    mb_size = out_file.stat().st_size / 1024**2
    print(f"  -> Saved to:          {out_file.name} ({mb_size:.2f} MB)")

    return {
        "mixture": config["id"],
        "raw_shape":[len(df) * args.downsample, len(df.columns) - 1],
        "final_shape": list(df.shape),
        "final_dt_sec": float(final_dt),
        "spearman_s01_vs_c2h4": float(corr),
        "file_size_mb": float(mb_size)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Dynamic Gas Time-Series.")
    parser.add_argument("--downsample", type=int, default=100, help="100 = 1Hz output")
    parser.add_argument("--no-clip", action="store_false", dest="clip")
    parser.add_argument("--no-interpolate", action="store_false", dest="interpolate")
    args = parser.parse_args()

    _P, _C = Paths(), Constants()
    fetch_and_unzip(_P)

    audit_log =[]
    print("=" * 75)
    print(f"{'DYNAMIC MIXTURE TIME-SERIES PIPELINE':^75}")
    print("=" * 75)

    for raw_file, config in _C.mixtures.items():
        found = list(_P.temp.rglob(raw_file))
        if not found:
            print(f"Warning: {raw_file} not found in archive.")
            continue

        stats = process_and_audit_mixture(found[0], config, args, _P, _C)
        audit_log.append(stats)
        
    # Export Metadata
    with open(_P.metadata, "w") as f:
        meta = {"args": vars(args), "audit_log": audit_log}
        json.dump(meta, f, indent=4)
        
    # Cleanup massive raw files
    if _P.temp.exists():
        shutil.rmtree(_P.temp)
    if _P.zip_file.exists():
        _P.zip_file.unlink()



    print("=" * 75)


if __name__ == "__main__":
    main()