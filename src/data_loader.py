import logging
import pickle
import re
from pathlib import Path

import mne
import numpy as np
import scipy.io

logger = logging.getLogger(__name__)


def load_seed_file(file_path, preload=True):
    """
    加载 SEED 数据集 .mat 文件。
    SEED 数据通常有 62 个通道，降采样至 200Hz。
    .mat 文件包含 15 个试验（键名如 *_eeg1, *_eeg2, ...）。
    """
    mat = scipy.io.loadmat(file_path)

    # 识别包含 EEG 数据的键
    # 键名通常如 'djc_eeg1', 'djc_eeg2' 等
    eeg_keys = [k for k in mat.keys() if "_eeg" in k]

    def get_eeg_index(key):
        match = re.search(r"eeg(\d+)", key)
        return int(match.group(1)) if match else 0

    eeg_keys.sort(key=get_eeg_index)

    if not eeg_keys:
        raise ValueError(
            f"No EEG data keys found in {file_path} (expected keys containing '_eeg')"
        )

    # 尝试加载 SEED 标签 (label.mat)
    labels = None
    # 假设 label.mat 在同一目录或父目录
    possible_label_paths = [
        Path(file_path).parent / "label.mat",
        Path(file_path).parent.parent / "label.mat",
    ]

    for lp in possible_label_paths:
        if lp.exists():
            try:
                l_mat = scipy.io.loadmat(lp)
                if "label" in l_mat:
                    labels = l_mat["label"]
                    logger.info(f"Loaded labels from {lp}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load labels from {lp}: {e}")

    # SEED 通道名称（62 个通道）
    ch_names = [
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "CB1",
        "O1",
        "OZ",
        "O2",
        "CB2",
    ]

    sfreq = 200  # SEED 通常是 200Hz
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # 连接所有试验以形成连续的 Raw 对象
    # 我们将添加注释来标记每个试验的开始
    all_data = []
    annotations = mne.Annotations(onset=[], duration=[], description=[])
    current_time = 0.0

    for key in eeg_keys:
        data = mat[key]  # 形状: (62, n_samples)
        n_samples = data.shape[1]
        duration = n_samples / sfreq

        all_data.append(data)

        annotations.append(current_time, duration, description=key)
        current_time += duration

    combined_data = np.concatenate(all_data, axis=1)

    raw = mne.io.RawArray(combined_data, info)
    raw.set_annotations(annotations)

    # 设置标准导联
    try:
        raw.set_montage("standard_1020")
    except Exception as e:
        logger.warning(f"Could not set montage: {e}")

    return raw, labels


def load_deap_file(file_path, preload=True):
    """
    加载 DEAP 数据集 .mat 或 .dat (pickle) 文件。
    DEAP 数据：32 名参与者，40 个视频。
    文件格式：'data' 键是 (40, 40, 8064)。(试验, 通道, 样本)
    通道：32 个 EEG + 8 个外周生理信号。
    """
    file_path = Path(file_path)
    if file_path.suffix == ".dat":
        with open(file_path, "rb") as f:
            content = pickle.load(f, encoding="latin1")
        data = content["data"]
        labels = content["labels"]
    else:
        mat = scipy.io.loadmat(file_path)

        if "data" not in mat:
            raise ValueError(f"Key 'data' not found in {file_path}")

        data = mat["data"]  # (40, 40, 8064)
        labels = mat.get("labels")  # (40, 4) - 效价, 唤醒度, 支配度, 喜爱度

    # DEAP 通道名称
    ch_names = [
        "Fp1",
        "AF3",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "P3",
        "P7",
        "PO3",
        "O1",
        "Oz",
        "Pz",
        "Fp2",
        "AF4",
        "Fz",
        "F4",
        "F8",
        "FC6",
        "FC2",
        "Cz",
        "C4",
        "T8",
        "CP6",
        "CP2",
        "P4",
        "P8",
        "PO4",
        "O2",
        "hEOG",
        "vEOG",
        "zEMG",
        "tEMG",
        "GSR",
        "Respiration",
        "Plethysmograph",
        "Temperature",
    ]

    ch_types = (
        ["eeg"] * 32 + ["eog"] * 2 + ["emg"] * 2 + ["gsr"] + ["resp"] + ["ch_type"] * 2
    )  # 简化类型
    # 修正 MNE 类型
    ch_types = ["eeg"] * 32 + [
        "eog",
        "eog",
        "emg",
        "emg",
        "gsr",
        "resp",
        "misc",
        "misc",
    ]

    sfreq = 128  # DEAP 降采样至 128Hz
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 连接试验
    n_trials, n_channels, n_samples = data.shape

    # 重塑为 (n_channels, n_trials * n_samples)
    # 但我们需要小心沿时间轴连接
    # data[i] 是 (40, 8064)

    combined_data = np.concatenate([data[i] for i in range(n_trials)], axis=1)

    raw = mne.io.RawArray(combined_data, info)

    # 添加注释
    duration = n_samples / sfreq
    onsets = np.arange(0, n_trials * duration, duration)
    descriptions = [f"Trial_{i + 1}" for i in range(n_trials)]

    annotations = mne.Annotations(
        onset=onsets, duration=[duration] * n_trials, description=descriptions
    )
    raw.set_annotations(annotations)

    try:
        raw.set_montage("standard_1020")
    except Exception as e:
        logger.warning(f"Could not set montage: {e}")

    return raw, labels


def load_generic_mat(file_path, preload=True):
    """
    Attempt to load a generic .mat file by finding the largest array.
    Assumes the data is EEG.
    """
    mat = scipy.io.loadmat(file_path)

    # Filter out internal keys (starting with __)
    valid_keys = [k for k in mat.keys() if not k.startswith("__")]

    if not valid_keys:
        raise ValueError(f"No valid keys found in {file_path}")

    # Find the largest array to assume it's the data
    best_key = None
    max_size = 0

    for key in valid_keys:
        val = mat[key]
        if isinstance(val, np.ndarray) and val.ndim in [2, 3]:
            if val.size > max_size:
                max_size = val.size
                best_key = key

    if best_key is None:
        raise ValueError("Could not find a suitable data array in .mat file.")

    data = mat[best_key]
    logger.info(
        f"Loading generic .mat data from key: '{best_key}' with shape {data.shape}"
    )

    labels = None
    # Handle structured array (MATLAB struct)
    if data.dtype.names:
        logger.info(f"Found structured array with fields: {data.dtype.names}")
        if "rawdata" in data.dtype.names:
            try:
                # Access the first element if it's a 1x1 struct
                if data.size == 1:
                    # data[0, 0] gives the void scalar, we access fields by name from the array or scalar
                    # For a (1,1) struct array 'data':
                    # data['rawdata'] is (1,1) object array containing the data
                    raw_val = data["rawdata"][0, 0]

                    if "label" in data.dtype.names:
                        labels = data["label"][0, 0]

                    data = raw_val
                    logger.info(
                        f"Extracted 'rawdata' from struct. New shape: {data.shape}"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract rawdata from struct: {e}")

    # Create dummy info
    sfreq = 250.0  # Default assumption

    n_trials = 0
    n_samples_per_trial = 0

    # Handle 3D data (Trials x Channels x Time) -> Concatenate to 2D
    if data.ndim == 3:
        # Assume shape is (n_trials, n_channels, n_samples)
        # We assume channels < samples usually
        s0, s1, s2 = data.shape
        if s1 < s2:
            # (Trials, Channels, Samples) -> (Channels, Trials*Samples)
            n_trials = s0
            n_samples_per_trial = s2
            data = np.concatenate([data[i] for i in range(s0)], axis=1)
        else:
            # Fallback: maybe (Trials, Samples, Channels)?
            # Try to rearrange to (Channels, Time)
            # This is a guess.
            n_trials = s0
            n_samples_per_trial = s1
            data = np.concatenate([data[i].T for i in range(s0)], axis=1)

    # Handle 2D data (Channels x Time) or (Time x Channels)
    if data.ndim == 2:
        n_rows, n_cols = data.shape
        # Assumption: Channels are fewer than time points
        if n_rows > n_cols:
            logger.info(
                "Transposing data assuming (Time, Channels) -> (Channels, Time)"
            )
            data = data.T

    n_channels = data.shape[0]

    ch_names = [f"Ch{i + 1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)

    # Add annotations if trials were detected
    if n_trials > 0 and n_samples_per_trial > 0:
        duration = n_samples_per_trial / sfreq
        onsets = np.arange(0, n_trials * duration, duration)
        descriptions = [f"Trial_{i + 1}" for i in range(n_trials)]
        annotations = mne.Annotations(
            onset=onsets, duration=[duration] * n_trials, description=descriptions
        )
        raw.set_annotations(annotations)
        logger.info(f"Added {n_trials} annotations based on 3D structure.")

    logger.warning(
        f"Loaded generic .mat file. Assumed sfreq={sfreq}Hz. "
        "Please update raw.info['sfreq'] and channel names manually if incorrect."
    )

    return raw, labels


def load_data(file_path, montage_name="standard_1020", preload=True):
    """
    从各种文件格式（EDF+, .mat, .gdf）加载 EEG 数据。

    参数:
        file_path (str or Path): EEG 文件的路径。
        montage_name (str): 要应用的导联名称（如果通道匹配）。
        preload (bool): 是否将数据预加载到内存中。

    返回:
        tuple: (mne.io.Raw, labels) 加载的原始 EEG 数据和对应的标签（如果存在）。
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    raw = None
    labels = None

    try:
        if extension == ".edf":
            logger.info(f"Loading EDF file: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=preload)
        elif extension == ".gdf":
            logger.info(f"Loading GDF file: {file_path}")
            raw = mne.io.read_raw_gdf(file_path, preload=preload)
        elif extension == ".bdf":
            logger.info(f"Loading BDF file: {file_path}")
            raw = mne.io.read_raw_bdf(file_path, preload=preload)
        elif extension == ".set":  # EEGLAB
            logger.info(f"Loading EEGLAB .set file: {file_path}")
            raw = mne.io.read_raw_eeglab(file_path, preload=preload)
        elif extension == ".dat":
            logger.info(f"Loading DEAP .dat file: {file_path}")
            raw, labels = load_deap_file(file_path, preload=preload)
        elif extension == ".mat":
            logger.info(f"Loading MAT file: {file_path}")
            # 检测 SEED 或 DEAP 的启发式方法
            try:
                mat_keys = scipy.io.whosmat(file_path)
                keys = [x[0] for x in mat_keys]

                if "data" in keys and "labels" in keys:
                    logger.info("Detected DEAP dataset structure.")
                    raw, labels = load_deap_file(file_path, preload=preload)
                elif any("_eeg" in k for k in keys):
                    logger.info("Detected SEED dataset structure.")
                    raw, labels = load_seed_file(file_path, preload=preload)
                else:
                    logger.info(
                        "SEED/DEAP structure not detected. Attempting generic load."
                    )
                    raw, labels = load_generic_mat(file_path, preload=preload)
            except Exception as e:
                logger.error(f"Failed to load .mat file: {e}")
                raise e
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        # 尝试从 Annotations 中提取标签 (对于非 .mat 文件)
        if labels is None and raw is not None:
            try:
                if len(raw.annotations) > 0:
                    events, _ = mne.events_from_annotations(raw)
                    labels = events[:, 2]
                    logger.info(f"Extracted {len(labels)} labels from annotations.")
            except Exception as e:
                logger.debug(f"Could not extract labels from annotations: {e}")

        # 如果可能，设置导联
        if montage_name and raw is not None:
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                # 仅为导联中存在的通道设置导联
                raw.set_montage(montage, match_case=False, on_missing="ignore")
            except Exception as e:
                logger.warning(f"Could not set montage: {e}")

        return raw, labels

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def find_dataset_files(data_dir, pattern=None):
    """
    Recursively find supported EEG files in a directory.

    Args:
        data_dir (str or Path): Directory to search.
        pattern (str, optional): Glob pattern to filter files.

    Returns:
        list: List of Path objects for supported files.
    """
    data_dir = Path(data_dir)
    supported_extensions = {".edf", ".gdf", ".bdf", ".set", ".mat", ".dat"}

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if pattern:
        # If pattern is provided, use it
        all_files = data_dir.rglob(pattern)
    else:
        # Otherwise, find all files
        all_files = data_dir.rglob("*")

    # Filter by extension
    return [
        f for f in all_files if f.is_file() and f.suffix.lower() in supported_extensions
    ]
