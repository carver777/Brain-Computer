import logging
import sys
from pathlib import Path

import mne
import numpy as np
import streamlit as st
import yaml

# 将 src 添加到路径
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_data
from src.features import compute_differential_entropy, compute_psd_features
from src.preprocessing import extract_epochs, preprocess_raw
from src.visualization import (
    plot_band_topomaps,
    plot_dimension_reduction,
    plot_feature_distribution,
)

# 配置日志记录以在 streamlit 中捕获（如果需要），或者仅抑制
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EEG 处理流程", layout="wide")

st.title("交互式 EEG 数据处理系统")

# --- 侧边栏配置 ---
st.sidebar.header("配置")

# 加载默认配置
config_path = Path("configs/config.yaml")
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)
else:
    st.error("未找到配置文件！")
    st.stop()

# 预处理设置
st.sidebar.subheader("预处理")
low_freq = st.sidebar.number_input(
    "低频截止 (Hz)", value=default_config["preprocessing"]["low_freq"]
)
high_freq = st.sidebar.number_input(
    "高频截止 (Hz)", value=default_config["preprocessing"]["high_freq"]
)
notch_freq = st.sidebar.number_input(
    "陷波滤波器 (Hz)", value=default_config["preprocessing"]["notch_freq"]
)
resample_freq = st.sidebar.number_input(
    "重采样率 (Hz)", value=default_config["preprocessing"]["resample_freq"]
)

# 分段设置
st.sidebar.subheader("分段")
tmin = st.sidebar.number_input("开始时间 (s)", value=default_config["epoching"]["tmin"])
tmax = st.sidebar.number_input("结束时间 (s)", value=default_config["epoching"]["tmax"])
overlap = st.sidebar.number_input(
    "重叠 (s)", value=default_config["epoching"]["overlap"]
)

# 特征设置
st.sidebar.subheader("特征 (PSD)")
fmin = st.sidebar.number_input(
    "PSD 最小频率", value=default_config["features"]["psd"]["fmin"]
)
fmax = st.sidebar.number_input(
    "PSD 最大频率", value=default_config["features"]["psd"]["fmax"]
)

# --- 主要内容 ---

# 文件选择
data_dir = Path("data")
if not data_dir.exists():
    st.error("未找到数据目录！")
    st.stop()

# 支持的文件扩展名
supported_extensions = [".edf", ".gdf", ".bdf", ".set", ".mat", ".dat"]

st.subheader("数据选择")

# Level 1: Dataset
datasets = [d for d in data_dir.iterdir() if d.is_dir()]
if not datasets:
    st.error("data/ 目录下没有找到数据集文件夹")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    selected_dataset = st.selectbox("数据集", datasets, format_func=lambda x: x.name)

# Level 2: Subject/Subfolder
children = list(selected_dataset.iterdir())
subdirs = sorted([d for d in children if d.is_dir()], key=lambda x: x.name)

selected_file_path = None

with col2:
    if subdirs:
        selected_subdir = st.selectbox(
            "受试者/子文件夹", subdirs, format_func=lambda x: x.name
        )
        # Level 3: File in Subdir
        subdir_files = sorted(
            [
                f
                for f in selected_subdir.iterdir()
                if f.is_file() and f.suffix in supported_extensions
            ],
            key=lambda x: x.name,
        )
        with col3:
            if subdir_files:
                selected_file = st.selectbox(
                    "文件", subdir_files, format_func=lambda x: x.name
                )
                selected_file_path = selected_file
            else:
                st.warning("该文件夹下无支持的文件")
    else:
        # No subdirs, look for files in dataset dir
        dataset_files = sorted(
            [f for f in children if f.is_file() and f.suffix in supported_extensions],
            key=lambda x: x.name,
        )
        with col3:
            if dataset_files:
                selected_file = st.selectbox(
                    "文件", dataset_files, format_func=lambda x: x.name
                )
                selected_file_path = selected_file
            else:
                st.warning("该数据集下无支持的文件")

if not selected_file_path:
    st.stop()

if st.button("处理数据"):
    with st.spinner("处理中..."):
        try:
            # 1. 加载数据
            st.info(f"正在加载 {selected_file_path.name}...")
            raw, gt_labels = load_data(selected_file_path)

            st.write(f"**通道数:** {len(raw.ch_names)}")
            st.write(f"**时间点:** {raw.n_times}")
            st.write(f"**采样率:** {raw.info['sfreq']} Hz")

            # 2. 预处理
            st.info("正在预处理...")
            raw_clean = preprocess_raw(
                raw,
                low_freq=low_freq,
                high_freq=high_freq,
                notch_freq=notch_freq,
                resample_freq=resample_freq,
            )

            # 显示原始数据与清洗后的数据（可选，也许只是一个片段）
            # st.subheader("原始数据预览")
            # st.pyplot(raw_clean.plot(show=False, duration=5, n_channels=10))

            # 3. 分段
            st.info("正在分段...")
            events, event_id = mne.events_from_annotations(raw_clean)

            if len(events) > 0:
                st.write(f"发现 {len(events)} 个事件。")
                epochs = extract_epochs(
                    raw_clean,
                    events,
                    event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=default_config["epoching"]["baseline"],
                )
            else:
                st.warning("未发现事件。使用固定长度分段。")
                epochs = mne.make_fixed_length_epochs(
                    raw_clean,
                    duration=tmax - tmin,
                    overlap=overlap,
                    preload=True,
                )

            st.write(f"**已创建分段:** {len(epochs)}")

            # 4. 特征提取
            st.info("正在提取特征...")
            bands = default_config["features"]["bands"]
            psds, freqs = compute_psd_features(epochs, fmin=fmin, fmax=fmax)
            de_features = compute_differential_entropy(psds, freqs, bands)

            st.write(f"**特征形状:** {de_features.shape}")

            # 5. 可视化
            st.subheader("可视化")

            # 特征分布
            st.write("### 特征分布")
            fig_dist = plot_feature_distribution(de_features)
            st.pyplot(fig_dist)

            # 准备标签
            labels = epochs.events[:, 2] if len(events) > 0 else np.zeros(len(epochs))
            plot_labels = gt_labels if gt_labels is not None else labels

            # 对齐标签逻辑 (与 main.py 保持一致)
            if gt_labels is not None:
                # 如果 epochs 被丢弃（例如由于伪影），我们需要对齐标签
                if len(epochs) < len(gt_labels) and len(gt_labels) == len(events):
                    st.info("Aligning ground truth labels with selected epochs.")
                    plot_labels = gt_labels[epochs.selection]
                # Handle case where epochs are sliding windows over trials (e.g. 40 trials -> 640 epochs)
                elif len(epochs) > len(gt_labels) and len(epochs) % len(gt_labels) == 0:
                    factor = len(epochs) // len(gt_labels)
                    st.info(f"Expanding labels by factor {factor} to match epochs.")
                    plot_labels = np.repeat(gt_labels, factor, axis=0)

            if (
                plot_labels is not None
                and plot_labels.ndim > 1
                and plot_labels.shape[0] > 1
            ):
                # 处理多维标签以进行可视化的简单逻辑
                valence = plot_labels[:, 0]
                plot_labels = (valence > 4.5).astype(int)
                st.info("使用二值化 Valence（阈值 4.5）作为可视化标签。")

            # 降维
            n_epochs, n_channels, n_bands = de_features.shape
            features_flat = de_features.reshape(n_epochs, -1)

            col1, col2 = st.columns(2)

            with col1:
                st.write("### t-SNE")
                try:
                    fig_tsne = plot_dimension_reduction(
                        features_flat, plot_labels, method="tsne"
                    )
                    st.pyplot(fig_tsne)
                except Exception as e:
                    st.error(f"t-SNE 失败: {e}")

            with col2:
                st.write("### PCA")
                try:
                    fig_pca = plot_dimension_reduction(
                        features_flat, plot_labels, method="pca"
                    )
                    st.pyplot(fig_pca)
                except Exception as e:
                    st.error(f"PCA 失败: {e}")

            # 地形图
            st.write("### 地形图")
            try:
                fig_topo = plot_band_topomaps(
                    de_features, raw_clean.ch_names, list(bands.keys())
                )
                st.pyplot(fig_topo)
            except Exception as e:
                st.error(f"地形图失败: {e}")

            st.success("处理完成！")

        except Exception as e:
            st.error(f"发生错误: {e}")
            logger.exception(e)
