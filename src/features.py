import numpy as np
from scipy.integrate import simpson


def compute_psd_features(epochs, fmin=1.0, fmax=50.0):
    """
    计算每个 epoch 的功率谱密度 (PSD) 特征。

    返回:
        psds: (n_epochs, n_channels, n_freqs)
        freqs: (n_freqs,)
    """
    psds, freqs = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax).get_data(
        return_freqs=True
    )
    return psds, freqs


def compute_band_power(psds, freqs, bands):
    """
    计算特定频带内的平均功率。

    参数:
        psds: (n_epochs, n_channels, n_freqs)
        freqs: (n_freqs,)
        bands: 字典 band_name -> (low_freq, high_freq)

    返回:
        band_powers: (n_epochs, n_channels, n_bands)
    """
    n_epochs, n_channels, n_freqs = psds.shape
    n_bands = len(bands)
    band_powers = np.zeros((n_epochs, n_channels, n_bands))

    for i, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        # 查找该频带内的频率索引
        idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)

        if np.sum(idx_band) == 0:
            continue

        # 该频带内的平均功率（积分近似）
        # 使用辛普森规则或简单平均。MNE 的 compute_psd 返回 power/Hz。
        # 对频率积分得到总功率。
        # 这里我们使用辛普森积分计算绝对频带功率。

        # 通过对 PSD 积分计算绝对频带功率
        freq_res = freqs[1] - freqs[0]
        bp = simpson(psds[:, :, idx_band], dx=freq_res, axis=-1)
        band_powers[:, :, i] = bp

    return band_powers


def compute_differential_entropy(psds, freqs, bands):
    """
    计算微分熵 (DE) 特征。
    对于 EEG，频带内的 DE 通常近似为 0.5 * log(2 * pi * e * variance)。
    如果我们假设频带内的信号是高斯的，DE 正比于 log(Energy)。
    这里我们将其近似为 log(Band Power)。

    返回:
        de_features: (n_epochs, n_channels, n_bands)
    """
    # 1. 首先计算频带功率
    band_powers = compute_band_power(psds, freqs, bands)

    # 2. 应用对数（处理零值/小值）
    # DE ~ log(Power)
    de_features = np.log(band_powers + 1e-10)

    return de_features


def normalize_features(features, method="zscore"):
    """
    对特征进行标准化。

    参数:
        features: (n_epochs, n_channels, n_bands)
        method: 'zscore' or 'minmax'

    返回:
        normalized_features: 形状同 features
    """
    if method == "zscore":
        # 沿 epoch 维度 (axis=0) 计算均值和标准差
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        return (features - mean) / (std + 1e-10)
    elif method == "minmax":
        min_val = np.min(features, axis=0, keepdims=True)
        max_val = np.max(features, axis=0, keepdims=True)
        return (features - min_val) / (max_val - min_val + 1e-10)
    else:
        return features
