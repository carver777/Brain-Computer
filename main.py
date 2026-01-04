import argparse
import logging
from pathlib import Path

import mne
import numpy as np
import yaml

from src.data_loader import find_dataset_files, load_data
from src.features import compute_differential_entropy, compute_psd_features
from src.preprocessing import extract_epochs, preprocess_raw
from src.visualization import (
    plot_band_topomaps,
    plot_dimension_reduction,
    plot_feature_distribution,
)


# 配置日志
def setup_logging(config):
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper())

    handlers = [logging.StreamHandler()]
    if log_config.get("save_to_file", False):
        log_file = log_config.get("log_file", "eeg_processing.log")
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="EEG Processing Pipeline")
    parser.add_argument(
        "--viz", action="store_true", help="Enable visualization of results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Filename pattern to filter datasets (e.g., 's01*', '*.mat')",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Relative path to the dataset directory within 'data/' to process.",
    )
    args = parser.parse_args()

    # 定义路径
    base_dir = Path(__file__).parent
    config_path = base_dir / args.config

    # 加载配置
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        return

    config = load_config(config_path)
    setup_logging(config)

    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

    # 确保目录存在
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    logger.info("Starting EEG processing pipeline...")

    # Determine search directory
    search_dir = data_dir

    # Interactive mode if no dataset argument is provided
    if not args.dataset:
        print("\n--- Available Datasets ---")
        datasets = [d for d in data_dir.iterdir() if d.is_dir()]

        if not datasets:
            print("No datasets found in 'data/' directory.")
            return

        for i, d in enumerate(datasets):
            print(f"{i + 1}. {d.name}")

        while True:
            try:
                choice = input(
                    "\nSelect a dataset number to process (or 'q' to quit): "
                )
                if choice.lower() == "q":
                    return
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    selected_dataset = datasets[idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        search_dir = selected_dataset
        print(f"\nSelected dataset: {selected_dataset.name}")

        # Try to find a real file to show in the tip
        sample_cmd = f'python tool/inspect_data.py "{search_dir}/<path_to_file>"'
        note = (
            "  (Replace <path_to_file> with an actual file inside the dataset folder)"
        )

        supported_exts = {".mat", ".edf", ".bdf", ".gdf", ".set", ".dat"}
        # Search for the first matching file
        found_files = [
            f
            for f in search_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_exts
        ]

        if found_files:
            try:
                # Try to make path relative to CWD for cleaner output
                rel_path = found_files[0].relative_to(Path.cwd())
                sample_cmd = f'python tool/inspect_data.py "{rel_path}"'
                note = ""
            except ValueError:
                sample_cmd = f'python tool/inspect_data.py "{found_files[0]}"'
                note = ""

        print(
            "\n[TIP] Before processing, you can inspect the data structure using the tool:"
        )
        print(f"  {sample_cmd}")
        if note:
            print(note)

        confirm = input(
            "\nDo you want to proceed with processing this dataset? (y/n): "
        )
        if confirm.lower() != "y":
            print("Processing aborted.")
            return

    elif args.dataset:
        search_dir = data_dir / args.dataset
        if not search_dir.exists():
            logger.error(f"Dataset directory not found: {search_dir}")
            return
        logger.info(f"Searching for data in: {search_dir}")

    try:
        files_to_process = find_dataset_files(search_dir, args.pattern)
    except Exception as e:
        logger.error(f"Error finding files: {e}")
        return

    if not files_to_process:
        logger.warning(
            f"No supported files found in {search_dir}. Please add .edf, .gdf, .set, or .mat files."
        )
        return

    # 定义频带
    bands = config["features"]["bands"]

    for file_path in files_to_process:
        try:
            logger.info(f"Processing file: {file_path.name}")

            # Determine dataset name and create output directories preserving hierarchy
            rel_path = file_path.relative_to(data_dir)
            dataset_root_name = rel_path.parts[0]

            # Subdirectory structure (if any) inside the dataset folder
            if len(rel_path.parts) > 2:
                sub_dirs = Path(*rel_path.parts[1:-1])
            else:
                sub_dirs = Path(".")

            dataset_results_dir = (
                results_dir / f"{dataset_root_name}-preprocessed-results" / sub_dirs
            )
            fif_dir = dataset_results_dir / "fif-data"
            npz_dir = dataset_results_dir / "npz-data"

            fif_dir.mkdir(parents=True, exist_ok=True)
            npz_dir.mkdir(parents=True, exist_ok=True)

            # 1. 加载数据
            raw, gt_labels = load_data(file_path)
            logger.info(
                f"Loaded data with {len(raw.ch_names)} channels and {raw.n_times} timepoints."
            )
            if gt_labels is not None:
                logger.info(f"Loaded ground truth labels with shape: {gt_labels.shape}")

            # 2. 预处理
            prep_cfg = config["preprocessing"]
            raw_clean = preprocess_raw(
                raw,
                low_freq=prep_cfg["low_freq"],
                high_freq=prep_cfg["high_freq"],
                notch_freq=prep_cfg["notch_freq"],
                resample_freq=prep_cfg["resample_freq"],
            )

            # 3. 保存预处理后的数据
            output_filename = fif_dir / f"{file_path.stem}_clean_raw.fif"
            raw_clean.save(output_filename, overwrite=True)
            logger.info(f"Saved processed data to {output_filename}")

            # 4. Epoching (分段)
            events, event_id = mne.events_from_annotations(raw_clean)
            epoch_cfg = config["epoching"]

            if len(events) > 0:
                logger.info(f"Found {len(events)} events. Extracting epochs...")
                # 假设我们提取事件后 1 秒的数据
                epochs = extract_epochs(
                    raw_clean,
                    events,
                    event_id,
                    tmin=epoch_cfg["tmin"],
                    tmax=epoch_cfg["tmax"],
                    baseline=epoch_cfg["baseline"],
                )
            else:
                logger.warning(
                    "No events found. Using fixed length epochs (1s sliding window)."
                )
                epochs = mne.make_fixed_length_epochs(
                    raw_clean,
                    duration=epoch_cfg["tmax"] - epoch_cfg["tmin"],
                    overlap=epoch_cfg["overlap"],
                    preload=True,
                )

            # 5. 特征提取
            logger.info("Extracting features (PSD and Differential Entropy)...")
            feat_cfg = config["features"]
            psds, freqs = compute_psd_features(
                epochs, fmin=feat_cfg["psd"]["fmin"], fmax=feat_cfg["psd"]["fmax"]
            )
            de_features = compute_differential_entropy(psds, freqs, bands)

            # de_features shape: (n_epochs, n_channels, n_bands)
            logger.info(f"Feature shape: {de_features.shape}")

            # 6. 保存特征
            feature_filename = npz_dir / f"{file_path.stem}_features.npz"

            labels = epochs.events[:, 2] if len(events) > 0 else np.zeros(len(epochs))

            if gt_labels is not None:
                # 如果 epochs 被丢弃（例如由于伪影），我们需要对齐标签
                if len(epochs) < len(gt_labels) and len(gt_labels) == len(events):
                    logger.info("Aligning ground truth labels with selected epochs.")
                    gt_labels = gt_labels[epochs.selection]
                # Handle case where epochs are sliding windows over trials (e.g. 40 trials -> 640 epochs)
                elif len(epochs) > len(gt_labels) and len(epochs) % len(gt_labels) == 0:
                    factor = len(epochs) // len(gt_labels)
                    logger.info(f"Expanding labels by factor {factor} to match epochs.")
                    gt_labels = np.repeat(gt_labels, factor, axis=0)

                np.savez(
                    feature_filename,
                    features=de_features,
                    labels=labels,
                    ch_names=raw_clean.ch_names,
                    bands=list(bands.keys()),
                    ground_truth_labels=gt_labels,
                )
            else:
                np.savez(
                    feature_filename,
                    features=de_features,
                    labels=labels,
                    ch_names=raw_clean.ch_names,
                    bands=list(bands.keys()),
                )
            logger.info(f"Saved features to {feature_filename}")

            # 7. 可视化 (如果启用)
            if args.viz or config["visualization"]["enable"]:
                logger.info("Generating visualizations...")
                viz_dir = dataset_results_dir / "plots" / file_path.stem
                viz_dir.mkdir(parents=True, exist_ok=True)

                # 绘制特征分布
                plot_feature_distribution(
                    de_features, save_path=viz_dir / "feature_distribution.png"
                )

                # 准备标签用于绘图
                plot_labels = gt_labels if gt_labels is not None else labels

                # 处理多维标签 (例如 DEAP 的 (n_samples, 4))
                if (
                    plot_labels is not None
                    and plot_labels.ndim > 1
                    and plot_labels.shape[0] > 1
                ):
                    logger.info(
                        "Detected multi-dimensional labels. Using the first dimension (Valence) for visualization."
                    )
                    # 取第一列 (Valence)
                    valence = plot_labels[:, 0]
                    # 二值化: > 4.5 为 High (1), <= 4.5 为 Low (0) (DEAP 范围 1-9, 中值约 5)
                    # 这里使用 4.5 作为阈值，或者你可以根据具体需求调整
                    plot_labels = (valence > 4.5).astype(int)
                    logger.info("Binarized labels based on Valence > 4.5.")

                # 展平特征用于降维
                n_epochs, n_channels, n_bands = de_features.shape
                features_flat = de_features.reshape(n_epochs, -1)

                # t-SNE
                try:
                    plot_dimension_reduction(
                        features_flat,
                        plot_labels,
                        method="tsne",
                        save_path=viz_dir / "tsne_visualization.png",
                    )
                except Exception as e:
                    logger.warning(f"Could not generate t-SNE: {e}")

                # PCA
                plot_dimension_reduction(
                    features_flat,
                    plot_labels,
                    method="pca",
                    save_path=viz_dir / "pca_visualization.png",
                )

                # Topomaps
                try:
                    plot_band_topomaps(
                        de_features,
                        raw_clean.ch_names,
                        list(bands.keys()),
                        save_path=viz_dir / "topomaps.png",
                    )
                except Exception as e:
                    logger.warning(f"Could not generate Topomaps: {e}")

                logger.info(f"Visualizations saved to {viz_dir}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    main()
