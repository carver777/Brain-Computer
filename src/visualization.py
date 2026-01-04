import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_feature_distribution(features, title="Feature Distribution", save_path=None):
    """
    绘制特征值的分布直方图。
    用于检查特征是否标准化，或是否有异常值。
    """
    plt.figure(figsize=(10, 6))
    # 展平特征以查看整体分布
    plt.hist(features.flatten(), bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("Feature Value (Differential Entropy)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        # plt.show()
        pass

    return plt.gcf()


def plot_dimension_reduction(features, labels, method="tsne", save_path=None):
    """
    使用 t-SNE 或 PCA 进行降维可视化。
    features: shape (n_samples, n_features)
    labels: shape (n_samples,)
    """
    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if method.lower() == "tsne":
        reducer = TSNE(
            n_components=2, random_state=42, init="pca", learning_rate="auto"
        )
        title = "t-SNE Visualization"
    else:
        reducer = PCA(n_components=2)
        title = "PCA Visualization"

    reduced_data = reducer.fit_transform(features_scaled)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # 使用 colormap
    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            color=colors[i],
            label=f"Class {label}",
            alpha=0.6,
            s=30,
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        # plt.show()
        pass

    return plt.gcf()


def plot_band_topomaps(features, ch_names, bands, info=None, save_path=None):
    """
    绘制每个频带的平均脑地形图。
    features: shape (n_samples, n_channels, n_bands)
    ch_names: list of channel names
    bands: list of band names
    """
    # 计算所有样本的平均值 -> (n_channels, n_bands)
    avg_features = np.mean(features, axis=0)

    n_bands = len(bands)

    # 如果没有提供 info，尝试创建一个标准的
    if info is None:
        # 创建一个 dummy info 对象用于绘图
        # 假设是标准的 10-20 系统
        montage = mne.channels.make_standard_montage("standard_1020")
        # 过滤掉不在 montage 中的通道
        valid_ch_names = [ch for ch in ch_names if ch in montage.ch_names]

        if len(valid_ch_names) < len(ch_names):
            print(
                f"Warning: {len(ch_names) - len(valid_ch_names)} channels not found in standard_1020 montage and will be ignored in topomap."
            )
            # 需要重新对齐数据
            valid_indices = [ch_names.index(ch) for ch in valid_ch_names]
            avg_features = avg_features[valid_indices, :]
            ch_names = valid_ch_names

        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
        info.set_montage(montage)

    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 5))
    if n_bands == 1:
        axes = [axes]

    for i, band in enumerate(bands):
        # 获取该频带的数据
        data = avg_features[:, i]

        im, _ = mne.viz.plot_topomap(
            data, info, axes=axes[i], show=False, cmap="RdBu_r", contours=0
        )
        axes[i].set_title(f"{band} Band")
        plt.colorbar(im, ax=axes[i], orientation="vertical", shrink=0.6)

    plt.suptitle("Average Differential Entropy Topomaps")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        # plt.show()
        pass

    return fig
