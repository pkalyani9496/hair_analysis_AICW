# AI Hair Intelligence System

An end-to-end multi-task deep learning platform for hair texture classification, scalp condition screening, semantic hair segmentation, salon haircut architectural recommendations, and in-memory virtual styling simulations.

---

## 🌟 Key Capabilities

1. **Hair Type Classification (PyTorch ResNet-18):**
   - Classifies hair into Straight, Wavy, Curly, and Kinky textures with calibrated softmax probabilities.
2. **Scalp Condition Screening (Calibrated ResNet-18 + Multi-Modal Verification):**
   - High-precision screening for scalp pathologies with calibrated decision thresholds.
   - Cross-validated against U-Net hair coverage to prevent facial skin false positives on healthy portraits.
3. **Semantic Hair Segmentation (Custom U-Net):**
   - Pixel-accurate binary hair mask extraction with morphological post-processing.
   - Anatomical region partitioning into Scalp, Middle, and Ends.
4. **Virtual Color Studio:**
   - 40+ curated shades (Natural Browns, Blondes, Vibrant Reds, and Creative Violets/Purples).
   - HSV color transfer preserving natural hair textures, highlights, and strand shadows.
5. **Salon Architecture Blueprints & Trimming Simulation:**
   - Interactive length and silhouette trimming simulation directly on user photos.
   - Architectural cutting horizons, layer tiers, and high-contrast segmented hair overlays.
6. **Zero-Download In-Memory Privacy Guarantee:**
   - All transformations and previews execute strictly in RAM; no user images are downloaded or saved to disk.

---

## 🚀 Quickstart & Local Installation

### 1. Clone & Set Up Environment
```bash
git clone <your-repo-url>
cd hair_analysis
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run hair_color_app.py
```
Open your browser at `http://localhost:8501`.

### 3. Optional: Run the FastAPI Backend
```bash
uvicorn backend.app:app --reload --port 8000
```
API documentation available at `http://localhost:8000/docs`.

---

## ☁️ Cloud Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended - Free & 1-Click)
1. Push this repository to your GitHub account.
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select this repository.
4. Set **Main file path** to `hair_color_app.py`.
5. Click **Deploy!**
*(Webcam and live camera inputs work seamlessly over Streamlit Cloud's default HTTPS!)*

### Option 2: Docker Container Deployment (Hugging Face Spaces, Render, Railway, AWS)
```bash
# Build the Docker image
docker build -t hair-intelligence-app .

# Run container locally on port 8501
docker run -p 8501:8501 hair-intelligence-app
```

---

## 🧪 Testing
Run the backend test suite:
```bash
pytest -v tests/test_backend.py
```

---

## 🔒 Privacy & Architecture
- **Inference Mode:** Optimized for CPU real-time execution.
- **Data Retention:** Zero persistent storage of user photographs.
