import logging
import sys
from pathlib import Path

import mne
import numpy as np
import streamlit as st
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_data
from src.features import compute_differential_entropy, compute_psd_features
from src.preprocessing import extract_epochs, preprocess_raw
from src.visualization import (
    plot_band_topomaps,
    plot_dimension_reduction,
    plot_feature_distribution,
)

# Configure logging to capture in streamlit if needed, or just suppress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EEG Processing Pipeline", layout="wide")

st.title("Interactive EEG Data Processing System")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Load default config
config_path = Path("configs/config.yaml")
if config_path.exists():
    with open(config_path, "r") as f:
        default_config = yaml.safe_load(f)
else:
    st.error("Config file not found!")
    st.stop()

# Preprocessing Settings
st.sidebar.subheader("Preprocessing")
low_freq = st.sidebar.number_input(
    "Low Cutoff (Hz)", value=default_config["preprocessing"]["low_freq"]
)
high_freq = st.sidebar.number_input(
    "High Cutoff (Hz)", value=default_config["preprocessing"]["high_freq"]
)
notch_freq = st.sidebar.number_input(
    "Notch Filter (Hz)", value=default_config["preprocessing"]["notch_freq"]
)
resample_freq = st.sidebar.number_input(
    "Resample Rate (Hz)", value=default_config["preprocessing"]["resample_freq"]
)

# Epoching Settings
st.sidebar.subheader("Epoching")
tmin = st.sidebar.number_input(
    "Start Time (s)", value=default_config["epoching"]["tmin"]
)
tmax = st.sidebar.number_input("End Time (s)", value=default_config["epoching"]["tmax"])
overlap = st.sidebar.number_input(
    "Overlap (s)", value=default_config["epoching"]["overlap"]
)

# Feature Settings
st.sidebar.subheader("Features (PSD)")
fmin = st.sidebar.number_input(
    "PSD Fmin", value=default_config["features"]["psd"]["fmin"]
)
fmax = st.sidebar.number_input(
    "PSD Fmax", value=default_config["features"]["psd"]["fmax"]
)

# --- Main Content ---

# File Selection
data_dir = Path("data")
if not data_dir.exists():
    st.error("Data directory not found!")
    st.stop()

# Recursive search for supported files
supported_extensions = [".edf", ".gdf", ".bdf", ".set", ".mat", ".dat"]
all_files = []
for ext in supported_extensions:
    all_files.extend(list(data_dir.rglob(f"*{ext}")))

file_options = [str(f.relative_to(data_dir)) for f in all_files]

if not file_options:
    st.warning("No supported EEG files found in 'data/' directory.")
    st.stop()

selected_file_str = st.selectbox("Select EEG File", file_options)
selected_file_path = data_dir / selected_file_str

if st.button("Process Data"):
    with st.spinner("Processing..."):
        try:
            # 1. Load Data
            st.info(f"Loading {selected_file_path.name}...")
            raw, gt_labels = load_data(selected_file_path)

            st.write(f"**Channels:** {len(raw.ch_names)}")
            st.write(f"**Timepoints:** {raw.n_times}")
            st.write(f"**Sampling Rate:** {raw.info['sfreq']} Hz")

            # 2. Preprocessing
            st.info("Preprocessing...")
            raw_clean = preprocess_raw(
                raw,
                low_freq=low_freq,
                high_freq=high_freq,
                notch_freq=notch_freq,
                resample_freq=resample_freq,
            )

            # Show raw vs clean (optional, maybe just a snippet)
            # st.subheader("Raw Data Preview")
            # st.pyplot(raw_clean.plot(show=False, duration=5, n_channels=10))

            # 3. Epoching
            st.info("Epoching...")
            events, event_id = mne.events_from_annotations(raw_clean)

            if len(events) > 0:
                st.write(f"Found {len(events)} events.")
                epochs = extract_epochs(
                    raw_clean,
                    events,
                    event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=default_config["epoching"]["baseline"],
                )
            else:
                st.warning("No events found. Using fixed length epochs.")
                epochs = mne.make_fixed_length_epochs(
                    raw_clean,
                    duration=tmax - tmin,
                    overlap=overlap,
                )

            st.write(f"**Epochs created:** {len(epochs)}")

            # 4. Feature Extraction
            st.info("Extracting Features...")
            bands = default_config["features"]["bands"]
            psds, freqs = compute_psd_features(epochs, fmin=fmin, fmax=fmax)
            de_features = compute_differential_entropy(psds, freqs, bands)

            st.write(f"**Feature Shape:** {de_features.shape}")

            # 5. Visualization
            st.subheader("Visualizations")

            # Feature Distribution
            st.write("### Feature Distribution")
            fig_dist = plot_feature_distribution(de_features)
            st.pyplot(fig_dist)

            # Prepare labels
            labels = epochs.events[:, 2] if len(events) > 0 else np.zeros(len(epochs))
            plot_labels = gt_labels if gt_labels is not None else labels

            if (
                plot_labels is not None
                and plot_labels.ndim > 1
                and plot_labels.shape[0] > 1
            ):
                # Simple logic to handle multi-dim labels for viz
                valence = plot_labels[:, 0]
                plot_labels = (valence > 4.5).astype(int)
                st.info(
                    "Using binarized Valence (threshold 4.5) for visualization labels."
                )

            # Dimension Reduction
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
                    st.error(f"t-SNE failed: {e}")

            with col2:
                st.write("### PCA")
                try:
                    fig_pca = plot_dimension_reduction(
                        features_flat, plot_labels, method="pca"
                    )
                    st.pyplot(fig_pca)
                except Exception as e:
                    st.error(f"PCA failed: {e}")

            # Topomaps
            st.write("### Topomaps")
            try:
                fig_topo = plot_band_topomaps(
                    de_features, raw_clean.ch_names, list(bands.keys())
                )
                st.pyplot(fig_topo)
            except Exception as e:
                st.error(f"Topomaps failed: {e}")

            st.success("Processing Complete!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(e)
