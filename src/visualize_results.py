import logging
import sys
from pathlib import Path

import numpy as np

# Add the project root to sys.path to allow imports from src when running this script directly
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization import (
    plot_band_topomaps,
    plot_dimension_reduction,
    plot_feature_distribution,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    base_dir = Path(__file__).parent.parent  # Point to project root
    results_dir = base_dir / "results"

    # 查找所有的 .npz 文件
    npz_files = list(results_dir.rglob("*_features.npz"))

    if not npz_files:
        logger.warning(
            f"No .npz files found in {results_dir}. Please run main.py first."
        )
        return

    for npz_file in npz_files:
        logger.info(f"Visualizing results for: {npz_file.name}")

        # 加载数据
        data = np.load(npz_file, allow_pickle=True)
        features = data["features"]  # (n_epochs, n_channels, n_bands)
        labels = data["labels"]
        ch_names = data["ch_names"].tolist()
        bands = data["bands"].tolist()

        # 如果有 ground_truth_labels，优先使用它作为标签
        if "ground_truth_labels" in data:
            plot_labels = data["ground_truth_labels"]
            logger.info("Using ground truth labels for visualization.")
        else:
            plot_labels = labels
            logger.info("Using event labels for visualization.")

        # 从父目录确定数据集名称
        # 检查新结构: results/<dataset>-preprocessed-results/npz-data/file.npz
        if npz_file.parent.name == "npz-data" and npz_file.parent.parent.name.endswith(
            "-preprocessed-results"
        ):
            viz_base = npz_file.parent.parent / "plots"
        # 检查旧结构: results/<dataset>-npz-data/file.npz
        elif npz_file.parent.name.endswith("-npz-data"):
            dataset_name = npz_file.parent.name.replace("-npz-data", "")
            viz_base = results_dir / f"{dataset_name}-plots"
        else:
            viz_base = results_dir / "plots"

        # 创建保存图像的目录
        viz_dir = viz_base / npz_file.stem
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 1. 绘制特征分布
        logger.info("Generating feature distribution plot...")
        plot_feature_distribution(
            features, save_path=viz_dir / "feature_distribution.png"
        )

        # 2. 绘制降维可视化 (t-SNE)
        # 需要将特征展平: (n_epochs, n_channels * n_bands)
        n_epochs, n_channels, n_bands = features.shape
        features_flat = features.reshape(n_epochs, -1)

        logger.info("Generating t-SNE plot...")
        try:
            plot_dimension_reduction(
                features_flat,
                plot_labels,
                method="tsne",
                save_path=viz_dir / "tsne_visualization.png",
            )
        except Exception as e:
            logger.error(f"Failed to generate t-SNE: {e}")

        # 3. 绘制 PCA
        logger.info("Generating PCA plot...")
        plot_dimension_reduction(
            features_flat,
            plot_labels,
            method="pca",
            save_path=viz_dir / "pca_visualization.png",
        )

        # 4. 绘制脑地形图
        logger.info("Generating Topomaps...")
        try:
            plot_band_topomaps(
                features, ch_names, bands, save_path=viz_dir / "topomaps.png"
            )
        except Exception as e:
            logger.error(f"Failed to generate Topomaps: {e}")

        logger.info(f"All plots saved to {viz_dir}")


if __name__ == "__main__":
    main()
