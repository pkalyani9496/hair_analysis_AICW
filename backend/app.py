from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.analysis_service import HairAnalysisService
from backend.recommendations import COLOR_PALETTE, HAIRCUT_RECOMMENDATIONS

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="AI Hair Intelligence API",
    version="1.0.0",
    description="Advanced backend for hair type, condition, segmentation and color analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = HairAnalysisService()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": "AI Hair Intelligence",
        "backend": "FastAPI",
        "version": "1.0.0",
    }


@app.get("/api/v1/overview")
def overview():
    return {
        "project": "AI Hair Intelligence",
        "features": [
            "Hair type classification",
            "Hair condition prediction",
            "Hair segmentation",
            "Color estimation",
            "Image quality scoring",
        ],
        "color_palette": [
            {"name": name, **details}
            for name, details in COLOR_PALETTE.items()
        ],
        "hair_types": list(HAIRCUT_RECOMMENDATIONS),
    }


@app.post("/api/v1/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    region: str | None = None,
    target_color: str | None = None,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type and "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    try:
        result = service.analyze_image(image, region=region or "Whole Hair", target_color=target_color)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}") from exc

    return result


@app.get("/api/v1/models")
def model_status():
    return {
        "status": "available",
        "models": {
            "hair_type": True,
            "disease": True,
            "segmentation": True,
        },
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
