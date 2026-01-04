# EEG 数据处理与分析平台 (EEG Data Processing & Analysis Platform)

本项目提供了一个通用的 EEG 数据处理流程，集成了数据加载、预处理、特征提取和可视化功能。支持多种标准数据格式及 SEED、DEAP 等公开数据集。提供命令行（CLI）批量处理和 Streamlit 交互式 Web 界面。

## ✨ 功能特性

- **多格式支持**:
  - 标准格式: `.edf`, `.gdf`, `.bdf`, `.set` (EEGLAB)
  - 公开数据集: SEED (`.mat`), DEAP (`.mat`, `.dat`)
  - 通用 `.mat` 文件自动识别
- **预处理流程**:
  - 滤波: 带通滤波 (Bandpass), 陷波滤波 (Notch)
  - 重采样 (Resampling)
  - 坏导修复 (插值)
  - 独立成分分析 (ICA) 去伪影 (配置中开启)
  - 重参考 (Re-referencing)
- **特征提取**:
  - 功率谱密度 (PSD)
  - 频带功率 (Band Power): Delta, Theta, Alpha, Beta, Gamma
  - 微分熵 (Differential Entropy, DE)
- **可视化**:
  - 脑地形图 (Topomaps)
  - 特征分布图
  - 降维可视化 (t-SNE, PCA)
- **交互式界面**: 基于 Streamlit 的 Web UI，支持实时参数调整和结果查看。

## 🛠️ 安装指南

确保已安装 Python 3.10+。

1. **克隆项目**

   ```bash
   git clone <repository_url>
   cd EEG
   ```

2. **安装依赖**
   本项目使用 `pyproject.toml` 管理依赖。

   ```bash
   pip install .
   uv sync
   ```

   或者直接安装主要依赖：

   ```bash
   pip install mne numpy scipy matplotlib pandas scikit-learn pyyaml streamlit pymatreader
   uv add mne numpy scipy matplotlib pandas scikit-learn pyyaml streamlit pymatreader
   ```

## 🚀 使用说明

### 1. 数据准备

将您的 EEG 数据放入 `data/` 目录中。建议按数据集或受试者组织文件夹，例如：

```txt
data/
├── SEED/
│   ├── 1_20131027.mat
│   └── ...
├── DEAP/
│   ├── s01.dat
│   └── ...
└── MyExperiment/
    ├── sub-01.edf
    └── ...
```

### 2. 交互式 Web 界面 (推荐)

启动 Streamlit 应用，在浏览器中进行可视化操作：

```bash
uv run streamlit run app.py
```

在界面侧边栏中，您可以调整预处理参数（滤波频率、分段时长等）并选择要分析的数据文件。

### 3. 命令行 (CLI) 批量处理

使用 `main.py` 进行自动化处理：

```bash
# 处理 data 目录下的所有支持文件
python main.py
uv run main.py

# 指定配置文件
python main.py --config configs/config.yaml
uv run main.py --config configs/config.yaml

# 过滤特定文件 (例如只处理 s01 开头的文件)
python main.py --pattern "s01*"
uv run main.py --pattern "s01*"

# 指定特定数据集子目录
python main.py --dataset "EEG datasets of stroke patients"
uv run main.py --dataset "EEG datasets of stroke patients"
```

## ⚙️ 配置说明

所有处理参数均在 `configs/config.yaml` 中定义。您可以修改此文件以适应不同的实验需求。

```yaml
preprocessing:
  low_freq: 1.0       # 低频截止
  high_freq: 40.0     # 高频截止
  notch_freq: 50.0    # 工频陷波 (50Hz 或 60Hz)
  resample_freq: 250.0 # 重采样率
  interpolation:
    enable: true      # 启用坏导插值
    method: "spline"

features:
  bands:              # 频带定义
    Delta: [1, 4]
    Theta: [4, 8]
    Alpha: [8, 13]
    Beta: [13, 30]
    Gamma: [30, 50]
```

## 📂 项目结构

```text
.
├── app.py                  # Streamlit Web 应用入口
├── main.py                 # 命令行处理入口
├── pyproject.toml          # 项目依赖配置
├── configs/
│   └── config.yaml         # 全局配置文件
├── data/                   # 数据输入目录
├── results/                # 处理结果输出目录 (预处理数据、图像)
└── src/                    # 核心代码库
    ├── data_loader.py      # 数据加载 (支持多种格式)
    ├── preprocessing.py    # 预处理算法 (滤波、插值等)
    ├── features.py         # 特征提取 (PSD, DE)
    └── visualization.py    # 绘图功能
```
