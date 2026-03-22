"""
Gas Sensor Array Drift Dataset Preparation & Audit.

Reference:
    Vergara et al. (2012) & Rodriguez-Lujan et al. (2014)
    DOI: 10.24432/C5MK6M
    URL: https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

The Machine Learning Task:
    This is a Concept Drift / Domain Generalization task, NOT a Time Series 
    forecasting problem. The high-resolution 100Hz temporal dynamics of each 
    gas exposure have already been flattened into 128 static features (Steady 
    State & EMA transients). Each row is an independent cross-sectional reading.
    
    The goal is to train a model (Classification for gas identity, or Regression 
    for concentration) on freshly calibrated hardware (Batches 1-6, representing 
    months 1-20), and evaluate its robustness on aged, degrading hardware 
    (Batches 7-10, representing months 21-36).

Preprocessing Issues Resolved:
    1. UCI Library Failure: The `ucimlrepo` library fails to parse the semicolon 
       separating `gas_id` and `concentration`, causing cascading NaNs.
    2. Flawed Community Fixes: Third-party scripts often duplicate rows to match 
       rounded documentation counts, destroying the domain sequence and causing 
       data leakage. We enforce strict immutability.
    3. Feature Opacity: Raw data uses generic indices (1-128). We map these to 
       physical hardware models (TGS2600, etc.) per the Lujan 2014 paper.
    4. Non-Linear Physics: We use Spearman Rank Correlation to validate the 
       Clifford-Tuma power-law model, proving signal maps to concentration.

Run Example:
    python prepare_gas_drift_dataset.py --data-dir ./OG --out-dir ./processed

"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class Paths:
    """Centralized path management."""

    def __init__(self, data_dir: str, out_dir: str) -> None:
        self.data_dir = pathlib.Path(data_dir)
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.out_dir / "gas_drift_ml.parquet"
        self.meta_path = self.out_dir / "gas_drift_metadata.json"


class Constants:
    """Centralized physical and experimental constants."""

    def __init__(self) -> None:
        self.target_instances = 13910
        self.target_features = 128
        self.gas_map = {
            1: "Ethanol",
            2: "Ethylene",
            3: "Ammonia",
            4: "Acetaldehyde",
            5: "Acetone",
            6: "Toluene",
        }
        self.sensor_models =["TGS2600", "TGS2602", "TGS2610", "TGS2620"]
        self.feature_types =[
            "dr",
            "dr_norm",
            "ema_0.001_rise",
            "ema_0.01_rise",
            "ema_0.1_rise",
            "ema_0.001_decay",
            "ema_0.01_decay",
            "ema_0.1_decay",
        ]


def generate_feature_columns(_C: Constants) -> list[str]:
    """Generates 128 descriptive column names based on the physical array."""
    # 4 sensors per model
    layout =[m for m in _C.sensor_models for _ in range(4)]
    return[
        f"s{i+1:02d}_{m.lower()}_{f}" for i, m in enumerate(layout) for f in _C.feature_types
    ]


def parse_batch_files(_P: Paths, _C: Constants) -> pd.DataFrame:
    """Parses the raw .dat files, repairing the semicolon and LibSVM issues."""
    all_rows =[]
    sensor_cols = generate_feature_columns(_C)

    for b in range(1, 11):
        file_path = _P.data_dir / f"batch{b}.dat"
        if not file_path.exists():
            print(f"Warning: {file_path.name} not found. Skipping.")
            continue

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                # Fix UCI semicolon issue: "label;concentration"
                meta_split = parts[0].split(";")
                gas_id, conc = int(meta_split[0]), float(meta_split[1])

                # Dense reconstruction of 128 features (0.0 padded)
                features = [0.0] * _C.target_features
                for p in parts[1:]:
                    if ":" in p:
                        idx_str, val_str = p.split(":")
                        idx = int(idx_str)
                        if 1 <= idx <= _C.target_features:
                            features[idx - 1] = float(val_str)

                all_rows.append([b, gas_id, _C.gas_map[gas_id], conc] + features)

    columns =["batch", "gas_id", "gas_name", "conc_ppmv"] + sensor_cols
    df = pd.DataFrame(all_rows, columns=columns)
    df["sensor_age"] = np.where(df["batch"] <= 6, "early", "aged")
    return df


def perform_scientific_audit(df: pd.DataFrame, _C: Constants) -> dict[str, any]:
    """Runs a strict suite of physical, temporal, and statistical validations."""
    print("\n" + "=" * 80)
    print(f"{'CRITICAL SCIENTIFIC AUDIT':^80}")
    print("=" * 80)

    audit_results = {}

    # 1. Completeness
    row_count = len(df)
    row_check = row_count == _C.target_instances
    print(f"1. Instance Count: {row_count}/13910 -> {'✅' if row_check else '❌'}")
    audit_results["completeness_verified"] = bool(row_check)

    # 2. Temporal Concept Drift (Toluene Sparsity)
    toluene_batches = df[df["gas_name"] == "Toluene"]["batch"].unique()
    sparsity_check = not set([3, 4, 5]).intersection(toluene_batches)
    print(f"2. Toluene Sparsity (Months 11-16 Empty): {'✅' if sparsity_check else '❌'}")
    audit_results["temporal_sparsity_verified"] = bool(sparsity_check)

    # 3. Physical Sign Logic
    dr_mean = float(df["s01_tgs2600_dr"].mean())
    decay_mean = float(df["s01_tgs2600_ema_0.1_decay"].mean())
    sign_check = dr_mean > 0 and decay_mean < 0
    print(f"3. Physical Transients (DR > 0, Decay < 0): {'✅' if sign_check else '❌'}")
    audit_results["physical_signs_verified"] = bool(sign_check)

    # 4. Non-Linear Monotonicity (Clifford-Tuma Power Law)
    ethanol_b1 = df[(df["gas_name"] == "Ethanol") & (df["batch"] == 1)]
    corr, _ = spearmanr(ethanol_b1["conc_ppmv"], ethanol_b1["s01_tgs2600_dr"])
    mono_check = corr > 0.85
    print(f"4. Spearman Monotonicity (Corr: {corr:.4f} > 0.85): {'✅' if mono_check else '❌'}")
    audit_results["monotonicity_spearman_corr"] = float(corr)

    # 5. Gas Distribution Stats
    print("\n5. Class Distribution & Concentration Ranges:")
    class_stats = {}
    for gas in _C.gas_map.values():
        gas_df = df[df["gas_name"] == gas]
        count = int(len(gas_df))
        c_min = float(gas_df["conc_ppmv"].min())
        c_max = float(gas_df["conc_ppmv"].max())
        print(f"   - {gas:<12}: {count:>4} rows | Range: {c_min:>5.1f} - {c_max:>6.1f} ppmv")
        class_stats[gas] = {"count": count, "min_ppmv": c_min, "max_ppmv": c_max}
    
    audit_results["class_statistics"] = class_stats
    print("=" * 80)
    
    return audit_results


def export_assets(df: pd.DataFrame, audit_results: dict[str, any], _P: Paths, _C: Constants) -> None:
    """Saves the ML-ready Parquet and the comprehensive physical JSON metadata."""
    df.to_parquet(_P.parquet_path, index=False)

    metadata = {
        "dataset_name": "Gas Sensor Array Drift at Different Concentrations",
        "doi": "10.24432/C5MK6M",
        "description": "13,910 measurements across 10 batches (36 months).",
        "citation": "Rodriguez-Lujan et al. (2014) Chemometrics and Int. Lab Systems.",
        "hardware_mapping": {
            "s01-s04": "TGS2600",
            "s05-s08": "TGS2602",
            "s09-s12": "TGS2610",
            "s13-s16": "TGS2620",
        },
        "feature_sequence": _C.feature_types,
        "class_mapping": _C.gas_map,
        "audit_results": audit_results,
    }

    with open(_P.meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSUCCESS: Assets exported to {_P.out_dir.absolute()}")

def make_ml_task(file_path: str | pathlib.Path) -> None:
    """Demonstrates how to load the dataset for Concept Drift ML tasks."""
    # 1. Load the ML-Ready dataset
    df = pd.read_parquet(file_path)

    # 2. Extract X (Features) and y (Targets)
    # Drop contextual metadata to isolate the 128 physical sensor readings.
    # Safely handles the "sensor_age" column if it was added during preprocessing.
    drop_cols =["batch", "gas_id", "gas_name", "conc_ppmv", "sensor_age"]
    existing_drop_cols =[c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=existing_drop_cols)
    y = df["gas_id"]

    # 3. Concept Drift Split (Train on Calibrated, Test on Degraded)
    # Batches 1-6 correspond to months 1-20 (Early/Calibrated life).
    # Batches 7-10 correspond to months 21-36 (Aged/Drifting life).
    train_mask = df["batch"] <= 6
    test_mask = df["batch"] >= 7

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Training shapes (Batches 1-6): X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes  (Batches 7-10): X={X_test.shape}, y={y_test.shape}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Process Gas Sensor Drift Dataset.")
    parser.add_argument("--data-dir", type=str, default="raw", help="Raw .dat directory")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    _P = Paths(data_dir=args.data_dir, out_dir=args.out_dir)
    _C = Constants()

    if not _P.data_dir.exists():
        print(f"Error: Data directory {_P.data_dir} does not exist.")
        return

    df = parse_batch_files(_P, _C)
    if df.empty:
        print("Error: No data parsed. Check your data directory.")
        return

    audit_results = perform_scientific_audit(df, _C)
    export_assets(df, audit_results, _P, _C)
    make_ml_task(_P.parquet_path)



if __name__ == "__main__":
    main()