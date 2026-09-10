# 🛰️ TempoSat — AI-Powered Satellite Image Temporal Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://temposat-app-jqft9f9xxvbz4y5sizwaep.streamlit.app/)

> Submitted to **Bharatiya Antariksh Hackathon (BAH) 2026** by ISRO — Problem Statement 12

---

## 🌍 What is TempoSat?

Satellites can only photograph the same location every 5–16 days.
TempoSat uses AI to **predict what a location looked like on the missing days**
between two real satellite captures — no extra satellites needed.

---

## 🚀 Live Demo

👉 **[Try it here](https://temposat-app-jqft9f9xxvbz4y5sizwaep.streamlit.app/)**

---

## ✨ Features

- 🤖 **Deep Residual U-Net** (7.70M params) trained from scratch on your images
- 🔬 **Self-supervised Super-Resolution** (256px → 1024px via RRDB/ESRGAN)
- 🗺️ **Google Earth Engine Integration** — download real Sentinel-2 data for any location
- 🌿 **NDVI Vegetation Health** mapping
- ☁️ **Cloud Detection** with coverage percentage
- 🔵 **Confidence Map** and Error Map
- 📊 **PSNR Performance Chart**
- ⬇️ **Download all outputs** as PNG files
- ⚡ **GPU-accelerated** training (30–45 seconds on RTX 3050)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch + CUDA (Mixed Precision FP16) |
| Web App | Streamlit |
| Satellite Data | Google Earth Engine + Sentinel-2 |
| Image Processing | OpenCV + Rasterio + NumPy |
| Map | Folium + streamlit-folium |

---

## 📦 Installation (Run Locally)

```bash
# 1. Clone the repository
git clone https://github.com/DhaneshSawant-coder/Temposat-app.git
cd TempoSat

# 2. Create conda environment
conda create -n temposat python=3.10 -y
conda activate temposat

# 3. Install GDAL and Rasterio first (via conda)
conda install -c conda-forge gdal rasterio -y

# 4. Install PyTorch with CUDA (adjust for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Authenticate Google Earth Engine
earthengine authenticate

# 7. Run the app
streamlit run app.py
```

---

## 🔑 Earth Engine Setup

For the Map Mode to work you need a free Google Earth Engine account:

1. Go to https://earthengine.google.com/ and sign up
2. Create a Google Cloud project
3. Enable the Earth Engine API
4. Run `earthengine authenticate` in your terminal

---

## 📁 Project Structure

TempoSat/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # This file
├── notebooks/ # Jupyter notebooks (development)
│ ├── 01_test_images.ipynb
│ ├── 02_ai_model.ipynb
│ ├── 03_final.ipynb
│ └── 04_real_satellite.ipynb
└── .gitignore
