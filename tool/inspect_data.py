import argparse
import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import scipy.io


def inspect_mat(file_path):
    print(f"Inspecting MAT file: {file_path}")
    try:
        mat = scipy.io.loadmat(file_path)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    print(f"Keys found: {list(mat.keys())}")

    valid_keys = [k for k in mat.keys() if not k.startswith("__")]

    print("\n--- Variable Analysis ---")
    for key in valid_keys:
        val = mat[key]
        print(f"Key: '{key}'")
        print(f"  Type: {type(val)}")
        if isinstance(val, np.ndarray):
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")
            if val.dtype.names:
                print(f"  Fields: {val.dtype.names}")
        print("-" * 20)

    print("\n--- Compatibility Check (src/data_loader.py) ---")

    # Check DEAP
    if "data" in valid_keys and "labels" in valid_keys:
        print(
            "[MATCH] DEAP dataset structure detected ('data' and 'labels' keys found)."
        )
        return

    # Check SEED
    eeg_keys = [k for k in valid_keys if "_eeg" in k]
    if eeg_keys:
        print(
            f"[MATCH] SEED dataset structure detected (Found keys with '_eeg': {eeg_keys})."
        )
        return

    # Check Generic/Stroke dataset (Struct with rawdata)
    best_key = None
    max_size = 0
    for key in valid_keys:
        val = mat[key]
        if isinstance(val, np.ndarray) and val.ndim in [2, 3]:
            if val.size > max_size:
                max_size = val.size
                best_key = key

    if best_key:
        print("[POSSIBLE MATCH] Generic/Stroke structure.")
        print(f"  Largest array found in key: '{best_key}'")
        val = mat[best_key]
        if val.dtype.names and "rawdata" in val.dtype.names:
            print("  [MATCH] Struct contains 'rawdata' field (Stroke dataset pattern).")
        else:
            print(
                "  [INFO] Standard array found. Data loader will attempt to load this."
            )
    else:
        print("[WARNING] No suitable data array found. Data loader might fail.")


def inspect_pickle(file_path):
    print(f"Inspecting Pickle file (.dat): {file_path}")
    try:
        with open(file_path, "rb") as f:
            content = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print(f"Type: {type(content)}")
    if isinstance(content, dict):
        print(f"Keys: {list(content.keys())}")
        if "data" in content and "labels" in content:
            print(
                "[MATCH] DEAP dataset structure detected (Pickle dict with 'data' and 'labels')."
            )
            print(f"  Data shape: {content['data'].shape}")
            print(f"  Labels shape: {content['labels'].shape}")
        else:
            print("[INFO] Dictionary found but keys do not match standard DEAP.")
    else:
        print("[INFO] File content is not a dictionary.")


def inspect_mne_supported(file_path):
    print(f"Inspecting MNE supported file: {file_path}")
    try:
        # Try to read info without loading data
        if file_path.suffix.lower() == ".edf":
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        elif file_path.suffix.lower() == ".bdf":
            raw = mne.io.read_raw_bdf(file_path, preload=False, verbose=False)
        elif file_path.suffix.lower() == ".gdf":
            raw = mne.io.read_raw_gdf(file_path, preload=False, verbose=False)
        elif file_path.suffix.lower() == ".set":
            raw = mne.io.read_raw_eeglab(file_path, preload=False, verbose=False)
        else:
            print(f"Unsupported extension for MNE inspection: {file_path.suffix}")
            return

        print("\n--- MNE Info ---")
        print(raw.info)
        print(f"\nChannels: {raw.ch_names}")
        print(f"Sampling Rate: {raw.info['sfreq']} Hz")
        print(f"Times: {raw.n_times} samples ({raw.times[-1]:.2f} s)")

        if raw.annotations:
            print(f"\nAnnotations: {len(raw.annotations)} found")
            print(raw.annotations.description)
        else:
            print("\nNo annotations found.")

        print("\n[MATCH] Successfully loaded with MNE.")

    except Exception as e:
        print(f"Error loading file with MNE: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect EEG data file structure.")
    parser.add_argument("file_path", type=str, help="Path to the data file")
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    ext = file_path.suffix.lower()

    if ext == ".mat":
        inspect_mat(file_path)
    elif ext == ".dat":
        # .dat can be pickle (DEAP) or BCI2000 (not handled here yet)
        inspect_pickle(file_path)
    elif ext in [".edf", ".bdf", ".gdf", ".set"]:
        inspect_mne_supported(file_path)
    else:
        print(f"Unknown extension: {ext}. Trying generic MNE load...")
        inspect_mne_supported(file_path)


if __name__ == "__main__":
    main()
