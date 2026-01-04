import logging

import mne

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


def extract_epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, baseline=(None, 0)):
    """
    基于事件从原始数据中提取 epochs。
    """
    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, baseline=baseline, preload=True
    )
    return epochs
