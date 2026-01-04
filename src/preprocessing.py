import logging

import mne
from mne.preprocessing import ICA

logger = logging.getLogger(__name__)


def preprocess_raw(
    raw, low_freq=1.0, high_freq=40.0, notch_freq=50.0, resample_freq=None
):
    """
    对原始 EEG 数据应用基本预处理。

    参数:
        raw (mne.io.Raw): 原始 EEG 数据。
        low_freq (float): 通带下边缘。
        high_freq (float): 通带上边缘。
        notch_freq (float or list): 陷波滤波器频率（例如 50Hz 或 60Hz）。
        resample_freq (float): 新的采样频率。

    返回:
        mne.io.Raw: 预处理后的原始数据。
    """
    raw = raw.copy()

    # 1. 滤波（带通）
    if low_freq is not None or high_freq is not None:
        logger.info(f"Applying bandpass filter: {low_freq}-{high_freq} Hz")
        raw.filter(l_freq=low_freq, h_freq=high_freq, fir_design="firwin")

    # 2. 陷波滤波（工频噪声）
    if notch_freq is not None:
        logger.info(f"Applying notch filter at {notch_freq} Hz")
        if isinstance(notch_freq, (int, float)):
            freqs = [notch_freq]
        else:
            freqs = notch_freq
        raw.notch_filter(freqs=freqs, fir_design="firwin")

    # 3. 重采样
    if resample_freq is not None:
        if raw.info["sfreq"] != resample_freq:
            logger.info(f"Resampling from {raw.info['sfreq']} to {resample_freq} Hz")
            raw.resample(resample_freq)

    return raw


def repair_channels(raw, method="spline", bad_channels=None):
    """
    修复坏导 (插值)。

    参数:
        raw (mne.io.Raw): 原始数据。
        method (str): 插值方法 ('spline' or 'mean').
        bad_channels (list): 手动指定的坏导名称列表。

    返回:
        mne.io.Raw: 修复后的数据。
    """
    raw = raw.copy()

    # 1. 标记手动指定的坏导
    if bad_channels:
        # 检查通道是否存在
        valid_bads = [ch for ch in bad_channels if ch in raw.ch_names]
        if valid_bads:
            logger.info(f"Marking manually specified bad channels: {valid_bads}")
            raw.info["bads"].extend(valid_bads)
            # 去重
            raw.info["bads"] = list(set(raw.info["bads"]))
        else:
            logger.warning(f"Specified bad channels {bad_channels} not found in data.")

    # 2. 简单的自动检测 (可选): 检测平坦通道
    # 这里我们简单地检查标准差是否极小
    # data = raw.get_data()
    # stds = np.std(data, axis=1)
    # flat_inds = np.where(stds < 1e-15)[0]
    # if len(flat_inds) > 0:
    #     flat_chs = [raw.ch_names[i] for i in flat_inds]
    #     logger.info(f"Auto-detected flat channels: {flat_chs}")
    #     raw.info['bads'].extend(flat_chs)

    # 3. 执行插值
    if raw.info["bads"]:
        logger.info(
            f"Interpolating {len(raw.info['bads'])} bad channels: {raw.info['bads']}"
        )
        try:
            raw.interpolate_bads(reset_bads=True, method=method, verbose=False)
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
    else:
        logger.info("No bad channels marked. Skipping interpolation.")

    return raw


def apply_ica(
    raw, n_components=20, method="fastica", random_state=97, exclude_eog=True
):
    """
    应用 ICA 去除伪影。

    参数:
        raw (mne.io.Raw): 原始数据。
        n_components (int): ICA 成分数量。
        method (str): ICA 方法 (例如 'fastica', 'infomax')。
        random_state (int): 随机种子。
        exclude_eog (bool): 是否尝试自动去除 EOG 伪影。

    返回:
        mne.io.Raw: 清洗后的数据。
    """
    logger.info(f"Fitting ICA with {n_components} components using {method}...")

    # 确保数据已加载到内存
    if not raw.preload:
        raw.load_data()

    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    ica.fit(raw)

    if exclude_eog:
        # 检查是否存在 EOG 通道
        if "eog" in raw.get_channel_types() or any(
            ch_type == "eog" for ch_type in raw.get_channel_types()
        ):
            logger.info("Attempting to find and remove EOG artifacts...")
            try:
                # find_bads_eog 会自动查找与 EOG 通道相关的成分
                eog_indices, eog_scores = ica.find_bads_eog(raw)
                if eog_indices:
                    logger.info(f"Found EOG artifacts indices: {eog_indices}")
                    ica.exclude = eog_indices
                else:
                    logger.info("No EOG artifacts found.")
            except Exception as e:
                logger.warning(f"Failed to find EOG artifacts: {e}")
        else:
            logger.warning(
                "No EOG channels found. Skipping automatic EOG artifact removal."
            )

    logger.info("Applying ICA to raw data...")
    raw = ica.apply(raw)
    return raw


def apply_rereference(raw, ref_channels="average"):
    """
    应用重参考 (Re-referencing)。
    默认使用 CAR (Common Average Reference)。

    参数:
        raw (mne.io.Raw): 原始数据。
        ref_channels (str or list): 参考通道。'average' 表示 CAR。

    返回:
        mne.io.Raw: 重参考后的数据。
    """
    logger.info(f"Applying re-referencing: {ref_channels}")
    raw_ref = raw.copy()
    raw_ref.set_eeg_reference(ref_channels=ref_channels, projection=False)
    return raw_ref


def extract_epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, baseline=(None, 0)):
    """
    基于事件从原始数据中提取 epochs。
    """
    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, baseline=baseline, preload=True
    )
    return epochs
