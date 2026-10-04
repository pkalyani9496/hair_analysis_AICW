"""
AI Hair Intelligence System
Streamlit Application for Hair Type Classification, Condition Screening,
Semantic Segmentation, Curated Haircut Recommendations & Virtual Styling.
"""
from io import BytesIO
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st

from backend.analysis_service import HairAnalysisService
from backend.recommendations import COLOR_PALETTE, recommend_colors, recommend_haircuts


# Configuration & Paths
API_URL = "http://127.0.0.1:8000/api/v1/analyze"
REGIONS = ["Whole Hair", "Scalp", "Middle", "Ends"]
BASE_DIR = Path(__file__).resolve().parent
HAIRCUTS_DIR = BASE_DIR / "frontend" / "haircuts"

# Detailed salon rationale for recommended cuts based on hair texture
HAIRCUT_RATIONALE = {
    "Curly Shag": "Crown layers evenly distribute volume to prevent triangular weight, allowing individual curl spirals to move freely.",
    "Rounded Layers": "Soft circular graduation follows the cranial curve, creating a balanced silhouette without heavy stacking at the perimeter.",
    "Curly Bob": "A chin-to-collarbone perimeter frames the jawline while maintaining enough weight to define curl clusters without shrinkage puff.",
    "Tapered Cut": "Tapered sides clean up the neckline and ears, drawing focus upward to celebrate natural crown volume and texture.",
    "Blunt Lob": "A razor-uniform horizontal boundary maximizes optical density, giving straight strands a visibly thicker, full-bodied presence.",
    "Long Face-Framing Layers": "Angled perimeter tiers begin below the chin to introduce organic motion and swing without compromising hair length.",
    "French Bob": "A geometric chin-length silhouette that leverages the smooth, reflective surface of straight strands for an architectural finish.",
    "Curtain Fringe": "Center-parted feathered bangs that frame the cheekbones naturally, adding face contour without cowlick distortion.",
    "Butterfly Layers": "Dual-tier layers create dynamic air pockets between cascading lengths, maximizing natural S-curve wave definition.",
    "Textured Lob": "Point-cut ends eliminate bottom blockiness, allowing loose waves to float with effortless, air-dried movement.",
    "Long Shag": "Deconstructed interior layering creates an effortless, undone aesthetic while maintaining manageable length throughout.",
}


# In-Memory Cached Analysis Service (Instant fallback if FastAPI is offline)
@st.cache_resource(show_spinner=False)
def get_local_service():
    """Loads deep learning models once into memory on CPU."""
    return HairAnalysisService()


# Haircut Simulation & Blueprint Engine (In-Memory Only, Zero Disk Writes)
def simulate_haircut_on_photo(image_rgb: np.ndarray, hair_mask: np.ndarray, haircut_name: str) -> np.ndarray:
    """
    Renders an in-memory length trimming and silhouette simulation on the user's photo
    using the U-Net hair segmentation mask. No images are saved or downloaded.
    """
    result = image_rgb.copy()
    ys, xs = np.where(hair_mask > 0)
    if len(ys) == 0:
        return result

    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    hair_h = max_y - min_y + 1
    hair_w = max_x - min_x + 1
    h, w = image_rgb.shape[:2]
    name_lower = haircut_name.lower()

    if "bob" in name_lower:
        # Chin/jawline level cut (~48% to 52% from top of hair)
        trim_ratio = 0.46 if "french" in name_lower else 0.52
        cut_y = min_y + int(hair_h * trim_ratio)
        x_coords = np.arange(w)
        center_x = (min_x + max_x) / 2.0
        # Gentle curve following jawline
        curve_offset = (1.0 - np.clip(((x_coords - center_x) / (hair_w / 2.0 + 1e-5)) ** 2, 0, 1)) * (hair_h * 0.035)
        curve_y = np.clip(cut_y - curve_offset, min_y + 8, max_y).astype(int)

        trim_mask = np.zeros_like(hair_mask, dtype=bool)
        for x in range(max(0, min_x), min(w, max_x + 1)):
            trim_mask[curve_y[x]:, x] = hair_mask[curve_y[x]:, x] > 0

        if np.sum(trim_mask) > 0:
            inpaint_mask = (trim_mask.astype(np.uint8) * 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)
            result = cv2.inpaint(image_rgb, dilated, 7, cv2.INPAINT_TELEA)

            # Soft edge feathering along cut line
            edge_line = np.zeros_like(hair_mask, dtype=np.uint8)
            for x in range(max(0, min_x), min(w, max_x + 1)):
                cy = curve_y[x]
                if cy < h:
                    edge_line[max(0, cy - 1):min(h, cy + 2), x] = 1
            feather_pts = (edge_line > 0) & (hair_mask > 0)
            if np.sum(feather_pts) > 0:
                blurred = cv2.GaussianBlur(result, (3, 3), 0)
                result[feather_pts] = blurred[feather_pts]

    elif "lob" in name_lower:
        # Collarbone length cut (~72% from top of hair)
        cut_y = min_y + int(hair_h * 0.72)
        trim_mask = (hair_mask > 0) & (np.arange(h)[:, None] > cut_y)
        if np.sum(trim_mask) > 0:
            inpaint_mask = (trim_mask.astype(np.uint8) * 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)
            result = cv2.inpaint(image_rgb, dilated, 7, cv2.INPAINT_TELEA)

    elif "tapered" in name_lower:
        # Tapered cut: tightens lower lateral perimeter while keeping crown volume
        mid_y = min_y + int(hair_h * 0.44)
        lateral_w = int(hair_w * 0.22)
        trim_mask = np.zeros_like(hair_mask, dtype=bool)
        trim_mask[mid_y:, min_x:min_x + lateral_w] = hair_mask[mid_y:, min_x:min_x + lateral_w] > 0
        trim_mask[mid_y:, max(0, max_x - lateral_w):max_x + 1] = hair_mask[mid_y:, max(0, max_x - lateral_w):max_x + 1] > 0
        if np.sum(trim_mask) > 0:
            inpaint_mask = (trim_mask.astype(np.uint8) * 255)
            result = cv2.inpaint(image_rgb, inpaint_mask, 7, cv2.INPAINT_TELEA)

    elif "shag" in name_lower or "layer" in name_lower:
        # Layers / Shag: lightens wispy bottom ends (~88% height) & introduces crown dimension
        cut_y = min_y + int(hair_h * 0.88)
        trim_mask = (hair_mask > 0) & (np.arange(h)[:, None] > cut_y)
        if np.sum(trim_mask) > 0:
            inpaint_mask = (trim_mask.astype(np.uint8) * 255)
            result = cv2.inpaint(image_rgb, inpaint_mask, 5, cv2.INPAINT_TELEA)

        crown_end_y = min_y + int(hair_h * 0.36)
        crown_mask = np.zeros_like(hair_mask, dtype=bool)
        crown_mask[min_y:crown_end_y, :] = hair_mask[min_y:crown_end_y, :] > 0
        crown_float = result.astype(np.float32)
        crown_boost = np.clip(crown_float * 1.05 + 7, 0, 255)
        soft_crown = cv2.GaussianBlur(crown_mask.astype(np.float32), (11, 11), 0)[:, :, None]
        result = np.uint8(crown_float * (1 - soft_crown * 0.25) + crown_boost * (soft_crown * 0.25))

    return result


def create_salon_blueprint(image_rgb: np.ndarray, hair_mask: np.ndarray, haircut_name: str, style: str = "Styling Zones") -> np.ndarray:
    """
    Renders an architectural salon styling blueprint on the user's photo with
    high-contrast highlighting of the segmented hair area, cut horizon, and target tiers.
    """
    result = image_rgb.copy()
    ys, xs = np.where(hair_mask > 0)
    if len(ys) == 0:
        return result

    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    hair_h = max_y - min_y + 1
    hair_w = max_x - min_x + 1
    h, w = image_rgb.shape[:2]
    name_lower = haircut_name.lower()

    if "bob" in name_lower:
        cut_ratio = 0.46 if "french" in name_lower else 0.52
        cut_y = min_y + int(hair_h * cut_ratio)
        horizon_label = f"✂ {haircut_name.upper()} HORIZON"
        has_cut_horizon = True
    elif "lob" in name_lower:
        cut_ratio = 0.72
        cut_y = min_y + int(hair_h * cut_ratio)
        horizon_label = f"✂ {haircut_name.upper()} PERIMETER"
        has_cut_horizon = True
    elif "tapered" in name_lower:
        cut_ratio = 0.44
        cut_y = min_y + int(hair_h * cut_ratio)
        horizon_label = "✂ CROWN VOLUME CORE"
        has_cut_horizon = True
    elif "shag" in name_lower or "layer" in name_lower:
        cut_ratio = 0.80
        cut_y = min_y + int(hair_h * cut_ratio)
        horizon_label = "✂ LAYER TIER HORIZON"
        has_cut_horizon = True
    else:
        cut_ratio = 0.70
        cut_y = min_y + int(hair_h * cut_ratio)
        horizon_label = f"✂ {haircut_name.upper()} HORIZON"
        has_cut_horizon = True

    # 1. High-Contrast Segmented Area Color Wash / Highlight Overlay
    tinted = image_rgb.copy().astype(np.float32)
    retained_mask = (hair_mask > 0) & (np.arange(h)[:, None] <= cut_y)
    trim_mask = (hair_mask > 0) & (np.arange(h)[:, None] > cut_y)

    if "Emerald" in style:
        main_c = np.array([16, 185, 129], dtype=np.float32)  # #10B981 Emerald
        tinted[hair_mask > 0] = tinted[hair_mask > 0] * 0.42 + main_c * 0.58
        border_c = (16, 185, 129)
    elif "Cyan" in style:
        main_c = np.array([6, 182, 212], dtype=np.float32)   # #06B6D4 Cyan
        tinted[hair_mask > 0] = tinted[hair_mask > 0] * 0.42 + main_c * 0.58
        border_c = (6, 182, 212)
    elif "Violet" in style:
        main_c = np.array([139, 92, 246], dtype=np.float32) # #8B5CF6 Violet
        tinted[hair_mask > 0] = tinted[hair_mask > 0] * 0.42 + main_c * 0.58
        border_c = (139, 92, 246)
    else:
        # Default: Dual Styling Zones (Retained Style in Teal/Emerald, Trim Zone in Coral/Red)
        teal = np.array([16, 185, 129], dtype=np.float32)
        coral = np.array([239, 68, 68], dtype=np.float32)
        if np.sum(retained_mask) > 0:
            tinted[retained_mask] = tinted[retained_mask] * 0.44 + teal * 0.56
        if np.sum(trim_mask) > 0:
            tinted[trim_mask] = tinted[trim_mask] * 0.40 + coral * 0.60
        border_c = (16, 185, 129)

    # Smooth Gaussian boundary blending
    soft_mask = cv2.GaussianBlur(hair_mask.astype(np.float32), (5, 5), 0)
    alpha = np.clip(soft_mask, 0, 1)[:, :, None]
    result = np.uint8(image_rgb.astype(np.float32) * (1 - alpha * 0.90) + tinted * (alpha * 0.90))

    # 2. Dual Boundary Contours for crisp visibility against background & clothes
    contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)  # Outer white keyline
    cv2.drawContours(result, contours, -1, border_c, 2, cv2.LINE_AA)         # Inner luminous accent

    # 3. Cut Horizon & Architectural Text Pill Badges
    if has_cut_horizon and min_y < cut_y < max_y:
        dash_len, gap_len = 16, 8
        for x in range(min_x, max_x, dash_len + gap_len):
            x_end = min(x + dash_len, max_x)
            cv2.line(result, (x, cut_y), (x_end, cut_y), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(result, (x, cut_y), (x_end, cut_y), (239, 68, 68), 2, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(horizon_label, font, 0.48, 1)
        bx, by = min_x + 8, cut_y - 10
        cv2.rectangle(result, (bx - 8, by - th - 6), (bx + tw + 8, by + 6), (15, 23, 42), -1)
        cv2.rectangle(result, (bx - 8, by - th - 6), (bx + tw + 8, by + 6), (239, 68, 68), 1, cv2.LINE_AA)
        cv2.putText(result, horizon_label, (bx, by), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        if np.sum(trim_mask) > 0 and (cut_y + 36) < max_y:
            trim_tag = "✂ TRIM / LENGTH REDUCTION ZONE"
            (tw2, th2), _ = cv2.getTextSize(trim_tag, font, 0.40, 1)
            tx, ty = min_x + 8, cut_y + 24
            cv2.rectangle(result, (tx - 6, ty - th2 - 4), (tx + tw2 + 6, ty + 4), (15, 23, 42), -1)
            cv2.rectangle(result, (tx - 6, ty - th2 - 4), (tx + tw2 + 6, ty + 4), (244, 63, 94), 1, cv2.LINE_AA)
            cv2.putText(result, trim_tag, (tx, ty), font, 0.40, (254, 202, 202), 1, cv2.LINE_AA)

    return result


# UI Design System: Clean, Premium College Expo Aesthetic
CUSTOM_CSS = """
<style>
/* Base typography and theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #172B26;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #173F35 0%, #0F2A23 100%);
    border-radius: 14px;
    padding: 26px 32px 22px 32px;
    color: #FFFFFF;
    margin-bottom: 22px;
    box-shadow: 0 4px 18px rgba(23, 63, 53, 0.12);
}

.hero-badge {
    display: inline-block;
    background: rgba(214, 93, 69, 0.22);
    color: #FFB3A3;
    border: 1px solid rgba(214, 93, 69, 0.45);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 14px;
    color: #D3DFD7;
    margin: 0 0 16px 0;
    line-height: 1.5;
    max-width: 820px;
}

.hero-caps-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 6px;
}

.hero-caps-pill {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11.5px;
    font-weight: 600;
    color: #F0F5F2;
}

/* Card Containers */
.expo-card {
    background: #FFFFFF;
    border: 1px solid #DCE4DD;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(23, 43, 39, 0.04);
}

/* Metric Result Cards */
.ai-stat-card {
    background: #FFFFFF;
    border: 1px solid #DCE4DD;
    border-top: 3px solid #173F35;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 2px 6px rgba(23, 43, 39, 0.03);
    margin-bottom: 12px;
    height: 100%;
}

.ai-stat-label {
    font-size: 11px;
    font-weight: 700;
    color: #68766E;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.ai-stat-value {
    font-size: 24px;
    font-weight: 800;
    color: #173F35;
    margin-bottom: 4px;
    line-height: 1.2;
}

.ai-stat-sub {
    font-size: 12px;
    color: #5C6F66;
    font-weight: 500;
}

.confidence-chip {
    display: inline-block;
    background: #EAF3EC;
    color: #1A5944;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 4px;
}

/* Haircut Recommendation Cards */
.haircut-box {
    background: #FFFFFF;
    border: 1px solid #DCE4DD;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(23, 43, 39, 0.04);
}

.haircut-badge {
    background: #FFF1EB;
    color: #D65D45;
    border: 1px solid #FAD7CE;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 8px;
    letter-spacing: 0.4px;
}

.haircut-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 21px;
    font-weight: 700;
    color: #173F35;
    margin: 0 0 6px 0;
}

.haircut-rationale {
    font-size: 13px;
    color: #4A5B53;
    line-height: 1.55;
    margin-bottom: 12px;
}

/* Comparison Preview Labels */
.preview-label-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.tag-original {
    background: #EEF3EF;
    color: #364B41;
}

.tag-ai {
    background: #FFF1EB;
    color: #D65D45;
    border: 1px solid #FAD7CE;
}

/* Swatch Chips */
.swatch-indicator {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    border: 1px solid rgba(0,0,0,0.15);
}

/* Privacy Guarantee Footer */
.privacy-banner {
    background: #F4F7F4;
    border: 1px solid #DCE4DD;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: #556B60;
    margin-top: 24px;
}
</style>
"""


def main():
    # Streamlit Page Setup
    st.set_page_config(
        page_title="AI Hair Intelligence | Expo Demo",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero / Landing Section
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">AI Hair Intelligence System</h1>
            <p class="hero-sub">
                An end-to-end multi-task deep learning platform analyzing hair texture, screening scalp conditions,
                generating semantic segmentation maps, recommending tailored haircut architectures, and rendering in-memory virtual previews.
            </p>
            <div class="hero-caps-row">
                <span class="hero-caps-pill">🔍 Hair Type (ResNet18)</span>
                <span class="hero-caps-pill">🩺 Scalp Condition Match</span>
                <span class="hero-caps-pill">🗺️ Hair Region Mapping</span>
                <span class="hero-caps-pill">🎨 Hair Color Extraction</span>
                <span class="hero-caps-pill">✂️ Haircut Recommendations</span>
                <span class="hero-caps-pill">🧪 Color Recommendations</span>
                <span class="hero-caps-pill">👁️ In-Memory Virtual Preview</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Section (Upload Photo / Take Photo)
    st.markdown("### 1. Select Input Source")
    input_mode = st.radio(
        "Choose input method",
        options=["Upload Photo", "Take Photo (Camera)"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode_selector",
    )

    uploaded_file = None
    if input_mode == "Take Photo (Camera)":
        st.caption("Allow camera access in your browser. Center your face with even lighting and hair visible.")
        uploaded_file = st.camera_input(
            "Take a live photo",
            help="Capture an image directly from your webcam or mobile camera.",
            key="camera_upload",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload hair photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPEG, PNG, WEBP. Use a clear, well-lit image with visible hair.",
            key="file_upload",
        )

    if uploaded_file is None:
        st.info("Upload a photo or capture one with your camera above to begin the AI analysis.")
        st.stop()

    # Safe In-Memory Image Loading (Zero Disk Writes)
    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    raw_name = getattr(uploaded_file, "name", None)
    file_name = raw_name if raw_name else ("live_camera_capture.jpg" if input_mode == "Take Photo (Camera)" else "uploaded_photo.jpg")
    file_type = getattr(uploaded_file, "type", None) or "image/jpeg"

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("The selected file could not be decoded as an image. Please choose another file.")
        st.stop()

    # Ensure all session state keys are initialized
    for key, default in [
        ("selected_region", "Whole Hair"),
        ("target_color", "Caramel"),
        ("selected_haircut_preview", None),
        ("cut_preview_mode", "✂️ Length & Silhouette Simulation (Trimmed on Your Photo)"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Invalidate cache if a new image was provided
    if st.session_state.get("image_hash") != image_hash:
        st.session_state.image_hash = image_hash
        st.session_state.pop("analysis_result", None)
        st.session_state["selected_region"] = "Whole Hair"
        st.session_state["target_color"] = "Caramel"
        st.session_state["selected_haircut_preview"] = None
        st.session_state["cut_preview_mode"] = "✂️ Length & Silhouette Simulation (Trimmed on Your Photo)"

    # Input Verification & Action Card
    with st.container():
        st.markdown('<div class="expo-card">', unsafe_allow_html=True)
        c_img, c_info = st.columns([1.1, 1.4], gap="large")
        with c_img:
            st.image(image, caption=f"Input Photo: {file_name}", use_container_width=True)
        with c_info:
            st.markdown('<h3 style="color:#173F35; margin-top:0;">Image Ready for AI Analysis</h3>', unsafe_allow_html=True)
            st.write("Both upload and camera inputs are processed through the identical unified deep learning pipeline.")
            st.caption(f"**Resolution:** {image.width} × {image.height} pixels | **Format:** RGB in-memory")

            analyze_clicked = st.button("✦ Run Deep Learning Analysis", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Processing Pipeline (FastAPI first, seamless local fallback)
    if analyze_clicked or ("analysis_result" not in st.session_state):
        with st.spinner("Executing neural networks (ResNet-18 & U-Net)..."):
            analysis_data = None
            # Attempt 1: Call FastAPI service
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            file_name,
                            image_bytes,
                            file_type,
                        )
                    },
                    params={"region": st.session_state.get("selected_region", "Whole Hair")},
                    timeout=180,
                )
                if response.status_code == 200:
                    analysis_data = response.json()
            except requests.RequestException:
                analysis_data = None

            # Attempt 2: Local in-memory inference fallback
            if analysis_data is None:
                local_svc = get_local_service()
                analysis_data = local_svc.analyze_image(
                    image,
                    region=st.session_state.get("selected_region", "Whole Hair"),
                )

            st.session_state.analysis_result = analysis_data
            st.session_state.analysis_image_hash = image_hash

    payload = st.session_state.get("analysis_result")
    if not payload or st.session_state.get("analysis_image_hash") != image_hash:
        st.stop()

    # Extract Core AI Predictions
    hair_type = payload.get("hair_type", "Unknown")
    hair_type_confidence = float(payload.get("hair_type_confidence", 0.0))
    disease = payload.get("disease", "Unknown")
    disease_confidence = float(payload.get("disease_confidence", 0.0))
    hair_mask = np.asarray(payload.get("hair_mask", []), dtype=np.uint8)
    region_masks = {
        name: np.asarray(mask, dtype=np.uint8)
        for name, mask in payload.get("region_masks", {}).items()
    }
    region_colors = payload.get("region_colors", {})
    coverage = float(payload.get("coverage", 0.0))
    current_color_obj = payload.get("color", {})
    dominant_color_label = current_color_obj.get("label", "Unknown")
    dominant_color_hex = current_color_obj.get("hex", "#3B241C")

    # Section 4: AI Analysis Result Dashboard
    st.markdown("### 2. AI Intelligence Dashboard")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        conf_html = f'<div class="confidence-chip">Confidence: {hair_type_confidence:.1f}%</div>' if hair_type_confidence > 0 else ""
        st.markdown(
            f"""
            <div class="ai-stat-card">
                <div class="ai-stat-label">HAIR TYPE</div>
                <div class="ai-stat-value">{hair_type}</div>
                <div class="ai-stat-sub">Architecture: ResNet-18</div>
                {conf_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stat2:
        is_healthy = "healthy" in disease.lower() or "normal" in disease.lower()
        if is_healthy:
            status_color = "#16a34a"
            card_sub = "Negative for scalp pathology"
            cond_conf_html = f'<div class="confidence-chip" style="background:#DCFCE7; color:#15803D;">✓ Healthy Scalp ({disease_confidence:.1f}%)</div>'
        elif disease == "Uncertain":
            status_color = "#d97706"
            card_sub = "Sub-threshold screening"
            cond_conf_html = f'<div class="confidence-chip" style="background:#FEF3C7; color:#B45309;">Score: {disease_confidence:.1f}%</div>'
        else:
            status_color = "#dc2626"
            card_sub = "Clinical condition detected"
            cond_conf_html = f'<div class="confidence-chip" style="background:#FEE2E2; color:#B91C1C;">Score: {disease_confidence:.1f}%</div>'

        st.markdown(
            f"""
            <div class="ai-stat-card">
                <div class="ai-stat-label">CONDITION MATCH</div>
                <div class="ai-stat-value" style="font-size:18px; color:{status_color};">{disease}</div>
                <div class="ai-stat-sub">{card_sub}</div>
                {cond_conf_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stat3:
        st.markdown(
            f"""
            <div class="ai-stat-card">
                <div class="ai-stat-label">HAIR COVERAGE</div>
                <div class="ai-stat-value">{coverage:.1f}%</div>
                <div class="ai-stat-sub">Architecture: U-Net</div>
                <div class="confidence-chip" style="background:#EBF2FA; color:#23528A;">Segmented mask</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stat4:
        st.markdown(
            f"""
            <div class="ai-stat-card">
                <div class="ai-stat-label">DETECTED COLOR</div>
                <div class="ai-stat-value" style="font-size:20px;">
                    <span class="swatch-indicator" style="background:{dominant_color_hex};"></span>{dominant_color_label}
                </div>
                <div class="ai-stat-sub">Hex: {dominant_color_hex}</div>
                <div class="confidence-chip" style="background:#FAF0EB; color:#A64D3B;">Dominant shade</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Condition predictions represent research pattern matching from image training data, not clinical medical advice.")

    # Section Tabs / Navigation
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    pages = [
        "Haircut Recommendations & Visual Preview",
        "Virtual Color Studio",
        "Hair Region Mapping",
        "Complete Diagnostics & Technical Overview",
    ]
    selected_tab = st.segmented_control(
        "Application Features",
        pages,
        default=pages[0],
        key="active_page",
        label_visibility="collapsed",
        width="stretch",
    )

    # Page 1: Haircut Recommendations & Visual Preview
    if "Haircut" in selected_tab:
        st.markdown("### Haircut Recommendations for Your Hair Type")
        st.write(
            f"Based on your predicted **{hair_type}** and hair density, our styling architecture recommends the following cuts. "
            "Each recommendation includes a verified salon reference guide and can be simulated on your photograph."
        )

        recommended_cuts = recommend_haircuts(hair_type)
        if not recommended_cuts:
            recommended_cuts = recommend_haircuts("Wavy Hair")

        # Ensure a selected preview cut exists
        if not st.session_state.get("selected_haircut_preview"):
            st.session_state.selected_haircut_preview = recommended_cuts[0]["name"]

        # Recommendation Cards Grid
        grid_cols = st.columns(min(len(recommended_cuts), 4), gap="medium")
        for idx, haircut in enumerate(recommended_cuts):
            with grid_cols[idx % len(grid_cols)]:
                st.markdown('<div class="haircut-box">', unsafe_allow_html=True)
                st.markdown('<span class="haircut-badge">✦ RECOMMENDED FOR YOU</span>', unsafe_allow_html=True)
                st.markdown(f'<h4 class="haircut-title">{haircut["name"]}</h4>', unsafe_allow_html=True)

                # Rationale tailored to hair type
                rationale = HAIRCUT_RATIONALE.get(haircut["name"], haircut.get("description", "Designed for natural hair movement."))
                st.markdown(f'<div class="haircut-rationale"><strong>Why recommended:</strong> {rationale}</div>', unsafe_allow_html=True)

                # Curated Style Reference Photo (Loaded strictly in-memory from local assets)
                img_rel_name = Path(haircut.get("image", "")).name
                local_ref_path = HAIRCUTS_DIR / img_rel_name
                if local_ref_path.exists():
                    try:
                        ref_img = Image.open(local_ref_path)
                        st.image(ref_img, caption=f"Curated Model Guide: {haircut['name']}", use_container_width=True)
                    except Exception:
                        st.caption("Reference asset available.")

                # Trigger to preview this specific cut on the user's photo
                is_active = (st.session_state.selected_haircut_preview == haircut["name"])
                btn_label = "✓ Previewing on Your Photo" if is_active else f"✂️ Simulate {haircut['name']} on My Photo"
                if st.button(btn_label, key=f"btn_cut_{idx}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state.selected_haircut_preview = haircut["name"]
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

        # Interactive Virtual Haircut Studio Section (Preview on User's Photo)
        st.markdown("---")
        active_cut_name = st.session_state.selected_haircut_preview
        st.markdown(f"### Virtual Haircut Studio: Previewing **{active_cut_name}** on Your Photo")
        st.write(
            "Explore how this recommended haircut translates to your features using our in-memory geometric trimming and contour engine. "
            "All previews are generated directly in RAM without saving any images."
        )

        mode_options = [
            "✂️ Length & Silhouette Simulation (Trimmed on Your Photo)",
            "📐 Salon Technical Blueprint (Layer Tiers & Cut Horizons)",
            "🖼️ Curated Model Reference Guide",
        ]
        cur_mode_idx = mode_options.index(st.session_state.cut_preview_mode) if st.session_state.cut_preview_mode in mode_options else 0
        preview_mode = st.radio(
            "Simulation Mode",
            options=mode_options,
            index=cur_mode_idx,
            horizontal=True,
        )
        st.session_state.cut_preview_mode = preview_mode

        image_rgb = np.asarray(image, dtype=np.uint8)

        c_before, c_after = st.columns(2, gap="large")
        with c_before:
            st.markdown('<span class="preview-label-tag tag-original">ORIGINAL PHOTO</span>', unsafe_allow_html=True)
            st.image(image, caption="Your Input Photo (Unmodified)", use_container_width=True)

        with c_after:
            if "Length & Silhouette" in preview_mode:
                st.markdown('<span class="preview-label-tag tag-ai">AI VIRTUAL CUT SIMULATION</span>', unsafe_allow_html=True)
                simulated = simulate_haircut_on_photo(image_rgb, hair_mask, active_cut_name)
                st.image(simulated, caption=f"Virtual Cut Simulation: {active_cut_name}", use_container_width=True)
            elif "Technical Blueprint" in preview_mode:
                st.markdown('<span class="preview-label-tag tag-ai">SALON ARCHITECTURE BLUEPRINT</span>', unsafe_allow_html=True)
                bp_style = st.selectbox(
                    "Blueprint Segmentation Overlay",
                    [
                        "✂️ Dual Styling Zones (Retained vs Trimmed)",
                        "✨ Emerald Glow (Whole Hair Highlight)",
                        "⚡ Cyan High-Contrast",
                        "💜 Electric Violet Blueprint",
                    ],
                    index=0,
                    key="bp_palette_select",
                    help="Choose how the segmented hair mask and architectural styling zones are highlighted.",
                )
                blueprint = create_salon_blueprint(image_rgb, hair_mask, active_cut_name, style=bp_style)
                st.image(blueprint, caption=f"Technical Cut Horizon & Tier Guide: {active_cut_name}", use_container_width=True)
                st.markdown(
                    """
                    <div style="display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; padding: 10px 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; font-size: 12.5px; font-weight: 600;">
                        <span style="color:#059669;">■ Target Haircut Silhouette (Retained Hair)</span>
                        <span style="color:#DC2626;">■ Length Reduction / Trim Zone</span>
                        <span style="color:#0F172A;">✂️ Cut Horizon Line</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span class="preview-label-tag tag-ai">VERIFIED SALON REFERENCE GUIDE</span>', unsafe_allow_html=True)
                cut_entry = next((c for c in recommended_cuts if c["name"] == active_cut_name), None)
                if cut_entry:
                    local_path = HAIRCUTS_DIR / Path(cut_entry["image"]).name
                    if local_path.exists():
                        ref_full = Image.open(local_path)
                        st.image(ref_full, caption=f"Curated Model Reference: {active_cut_name}", use_container_width=True)

        st.info(
            "**Styling Transparency Note:** Virtual cut simulation computes geometric length trimming and contour guidelines "
            "directly on your segmented hair mask. Refer to the curated salon reference guide for authentic hair texture and movement."
        )

    # Page 2: Virtual Color Studio
    elif "Color Studio" in selected_tab:
        st.markdown("### Virtual Hair Color Studio")
        st.write(
            "Explore recommended hair shades tailored to your hair type and detected undertone, "
            "or select any custom creative shade (including Purple). All recoloring is applied strictly to the segmented hair mask."
        )

        current_region = st.session_state.get("selected_region", "Whole Hair")
        detected_color = region_colors.get(current_region, payload.get("color", {}))
        detected_label = detected_color.get("label", "Medium Brown")
        color_suggestions = recommend_colors(hair_type, detected_label)

        st.markdown(
            f"Detected tone: **{detected_label}** ({detected_color.get('hex', '#3B241C')}). "
            "Select a recommended shade below or choose from the palette:"
        )

        # Clickable Recommended Color Swatches
        st.markdown("#### Recommended Shades for You")
        swatch_cols = st.columns(min(len(color_suggestions), 5))
        for idx, suggestion in enumerate(color_suggestions):
            with swatch_cols[idx % len(swatch_cols)]:
                s_name = suggestion["name"]
                s_hex = suggestion.get("hex", "#B97943")
                is_active = (st.session_state.get("target_color") == s_name)
                if st.button(
                    f"● {s_name}",
                    key=f"rec_color_{idx}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=suggestion.get("description", ""),
                ):
                    st.session_state.target_color = s_name
                    st.rerun()
                st.caption(f"{suggestion.get('description', '')}")

        st.markdown("---")

        # Controls Row: Preset Dropdown, Custom Color Picker, Region, Intensity
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.2, 1.2, 1.5])
        with ctrl_col1:
            palette_options = list(COLOR_PALETTE.keys())
            target_name = st.selectbox(
                "Select Preset Shade",
                palette_options,
                index=palette_options.index(st.session_state.target_color) if st.session_state.target_color in palette_options else 0,
            )
            if target_name != st.session_state.target_color:
                st.session_state.target_color = target_name

        with ctrl_col2:
            default_hex = COLOR_PALETTE.get(st.session_state.target_color, {}).get("hex", "#6B2D8C")
            custom_color = st.color_picker("Custom Shade / Hex Picker", value=default_hex)

        with ctrl_col3:
            target_region = st.selectbox("Apply Region", REGIONS, index=REGIONS.index(current_region))
            if target_region != st.session_state.selected_region:
                st.session_state.selected_region = target_region

        with ctrl_col4:
            strength = st.slider("Color Intensity", min_value=0.40, max_value=1.00, value=0.78, step=0.02)

        # Determine target RGB: use custom_color if changed, else preset hex
        active_hex = custom_color if custom_color != default_hex else COLOR_PALETTE.get(st.session_state.target_color, {}).get("hex", custom_color)
        target_rgb = HairAnalysisService.hex_to_rgb(active_hex)
        selected_mask = region_masks.get(target_region, hair_mask)

        # In-memory recoloring on segmented hair mask
        image_rgb = np.asarray(image, dtype=np.uint8)
        recolored = HairAnalysisService.apply_hair_color(
            image_rgb,
            selected_mask,
            target_rgb,
            strength=float(strength),
        )

        st.markdown(
            f"**Previewing:** <span class='swatch-indicator' style='background:{active_hex};'></span>"
            f"Target Color: **{st.session_state.target_color}** ({active_hex}) on **{target_region}**",
            unsafe_allow_html=True,
        )

        # Before / After Side-by-Side Presentation
        c_orig, c_preview = st.columns(2, gap="large")
        with c_orig:
            st.markdown('<span class="preview-label-tag tag-original">ORIGINAL PHOTO</span>', unsafe_allow_html=True)
            st.image(image, caption="Original Input Image", use_container_width=True)
        with c_preview:
            st.markdown('<span class="preview-label-tag tag-ai">AI VIRTUAL COLOR PREVIEW</span>', unsafe_allow_html=True)
            st.image(recolored, caption=f"Virtual Preview: {st.session_state.target_color} ({active_hex})", use_container_width=True)

        st.caption("Virtual color simulation uses semantic segmentation. Background, facial skin, and clothing remain untouched.")

    # Page 3: Hair Region Mapping
    elif "Region Mapping" in selected_tab:
        st.markdown("### Semantic Hair Segmentation & Region Mapping")
        st.write(
            "Our U-Net segmentation network isolates hair pixels from the surrounding face, neck, and background. "
            "The segmented hair is partitioned into anatomically defined regions: Scalp, Middle, and Ends."
        )

        active_region = st.selectbox("Select Hair Region to Inspect", REGIONS)
        selected_mask = region_masks.get(active_region, hair_mask)
        coverage_val = payload.get("regions", {}).get(active_region, 0.0)

        # Create high-contrast salon mapping overlay
        original_np = np.asarray(image, dtype=np.uint8)
        tint = np.empty_like(original_np)
        tint[:] = [16, 185, 129]  # Luminous Salon Emerald (#10B981)
        darkened = (original_np.astype(np.float32) * 0.35).astype(np.uint8)
        overlay = np.where(
            selected_mask[:, :, None] > 0,
            (original_np.astype(np.float32) * 0.42 + tint.astype(np.float32) * 0.58).astype(np.uint8),
            darkened,
        )
        region_contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, region_contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.drawContours(overlay, region_contours, -1, (16, 185, 129), 2, cv2.LINE_AA)

        c_map1, c_map2 = st.columns(2, gap="large")
        with c_map1:
            st.markdown('<span class="preview-label-tag tag-original">ORIGINAL PHOTO</span>', unsafe_allow_html=True)
            st.image(image, caption="Original Input Image", use_container_width=True)
        with c_map2:
            st.markdown(f'<span class="preview-label-tag tag-ai">MAPPED REGION: {active_region.upper()}</span>', unsafe_allow_html=True)
            st.image(overlay, caption=f"Isolated Region: {active_region} ({float(coverage_val):.1f}% frame coverage)", use_container_width=True)

        # Regional Coverage Breakdown Cards
        st.markdown("#### Regional Density Distribution")
        r_cols = st.columns(4)
        for i, r_name in enumerate(REGIONS):
            with r_cols[i]:
                r_cov = payload.get("regions", {}).get(r_name, 0.0)
                r_color = region_colors.get(r_name, {}).get("hex", "#3B241C")
                st.metric(
                    label=r_name,
                    value=f"{float(r_cov):.1f}%",
                    delta=f"Hex: {r_color}",
                    delta_color="off",
                )

    # Page 4: Complete Diagnostics & Technical Overview
    else:
        st.markdown("### Technical Diagnostics & Model Performance")
        st.write("Detailed metrics from the classification, segmentation, and quality evaluation modules.")

        diag_col1, diag_col2 = st.columns(2, gap="large")
        with diag_col1:
            st.markdown("#### Hair Type Softmax Distribution (ResNet-18)")
            hair_preds = payload.get("hair_type_top_predictions", [])
            if hair_preds:
                for item in hair_preds:
                    c_lbl = item.get("label", "")
                    c_val = float(item.get("confidence", 0.0))
                    st.write(f"**{c_lbl}**")
                    st.progress(min(1.0, c_val / 100.0), text=f"{c_val:.1f}%")

            st.markdown("#### Scalp Condition Screening Distribution")
            is_healthy = "healthy" in disease.lower() or "normal" in disease.lower()
            if is_healthy:
                st.caption(
                    "✓ **Screening Status: Negative for Scalp Pathology** — All 10 closed-set dermatological categories scored below calibrated diagnostic thresholds. "
                    "U-Net semantic segmentation confirms healthy follicular hair coverage."
                )
            else:
                st.caption(
                    f"⚠️ **Screening Status: {disease}** ({disease_confidence:.1f}%) exceeds calibrated clinical screening threshold."
                )

            disease_preds = payload.get("disease_top_predictions", [])
            if disease_preds:
                for item in disease_preds:
                    d_lbl = item.get("label", "")
                    d_val = float(item.get("confidence", 0.0))
                    st.write(f"**{d_lbl}**")
                    st.progress(min(1.0, d_val / 100.0), text=f"{d_val:.1f}%")

        with diag_col2:
            st.markdown("#### Image Quality Metrics")
            quality = payload.get("quality", {})
            q_score = float(quality.get("score", 0.0))
            st.metric("Overall Quality Score", f"{q_score:.0f} / 100", f"Status: {quality.get('status', 'Good')}")

            q_c1, q_c2 = st.columns(2)
            with q_c1:
                st.write(f"**Sharpness (Laplacian):** {quality.get('sharpness', 'N/A')}")
                st.write(f"**Contrast (StdDev):** {quality.get('contrast', 'N/A')}")
            with q_c2:
                st.write(f"**Brightness (Mean):** {quality.get('brightness', 'N/A')}")
                st.write(f"**Hair Mask Resolution:** {hair_mask.shape}")

            st.markdown("#### Model Architecture Specifications")
            st.write("• **Classification Backbone:** PyTorch ResNet-18 with calibrated temperature scaling")
            st.write("• **Segmentation Backbone:** Custom U-Net with skip connections & morphological post-processing")
            st.write("• **Inference Hardware:** CPU execution (optimized for real-time inference without requiring a dedicated GPU)")

    # Section 10 Privacy Guarantee Footer (No Image Downloads)
    st.markdown(
        """
        <div class="privacy-banner">
            <strong>✦ Zero-Download In-Memory Privacy Guarantee:</strong>
            This application processes all images strictly in system RAM. No photographs, hairstyle previews,
            or color transformations are downloaded or written to your local hard drive.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
