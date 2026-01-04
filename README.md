# EEG 处理流程

本项目提供了一个处理 EEG 数据的流程，支持标准格式（EDF, GDF, BDF, SET）和特定数据集（SEED, DEAP）。

## 支持的格式

该流程会自动检测并加载以下标准 EEG 文件格式：

- **.edf** (European Data Format)
- **.gdf** (General Data Format)
- **.bdf** (BioSemi Data Format)
- **.set** (EEGLAB Dataset)

对于这些格式，系统使用 `mne` 的标准读取器，并尝试应用标准的 10-20 导联系统。

## 支持的数据集（特殊处理）

### SEED 数据集

- **格式**: `.mat` 文件。
- **结构**: 包含如 `djc_eeg1`, `djc_eeg2` 等键。
- **处理**:
  - 加载 62 个 EEG 通道。
  - 将所有试验拼接成单个连续记录。
  - 添加试验边界的注释。
  - 采样率: 200 Hz。

### DEAP 数据集

- **格式**: `.mat` 文件。
- **结构**: 包含 `data` (40 试验 x 40 通道 x 8064 样本) 和 `labels`。
- **处理**:
  - 加载 32 个 EEG 通道 + 8 个外围通道。
  - 拼接所有 40 个试验。
  - 添加试验边界的注释。
  - 采样率: 128 Hz。

## 用法

1. 将数据文件放入 `data/` 目录。
2. **数据处理与特征提取**:
   运行主脚本以清洗数据并提取特征（PSD 和微分熵）：

   ```bash
   python main.py
   ```

   结果将保存在 `results/` 中：
   - `*_clean_raw.fif`: 预处理后的原始数据。
   - `*_features.npz`: 提取的特征和标签。

## 特征提取详情

项目目前提取以下特征：

- **功率谱密度 (PSD)**: 使用 Welch 方法计算。
- **微分熵 (Differential Entropy)**: 基于 5 个频带 (Delta, Theta, Alpha, Beta, Gamma) 的对数能量。

## 依赖项

- `mne`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
