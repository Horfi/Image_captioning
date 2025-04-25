import io
import time
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

from models.caption_model import CaptionModel
from utils.preprocess import preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Caption Generator API",
    version="1.0.0"
)

# CORS settings (adjust origins in production)
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Paths
this_dir = Path(__file__).resolve().parent
MODEL_PATH = this_dir / "models" / "caption_model_200.keras"

# Load model at startup
after_startup = None  # placeholder for linter
def load_model_on_startup():
    global caption_model
    caption_model = CaptionModel()
    if not MODEL_PATH.is_file():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    caption_model.load_model(str(MODEL_PATH))
    logger.info("Caption model loaded successfully")

app.add_event_handler("startup", load_model_on_startup)

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Image Caption Generator API is running"}

@app.post("/api/caption")
async def generate_caption(file: UploadFile = File(...)):
    # Validate content type
    if not file.content_type.startswith("image/"):
        logger.warning("Non-image upload: %s", file.filename)
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image bytes
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        logger.exception("Invalid image file uploaded")
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess
    try:
        np_img = preprocess_image(img)
    except Exception:
        logger.exception("Error during preprocessing")
        raise HTTPException(status_code=500, detail="Error preprocessing image")

    # Generate caption
    t0 = time.time()
    try:
        text = caption_model.generate_caption(np_img)
    except Exception:
        logger.exception("Error in model inference")
        raise HTTPException(status_code=500, detail="Error generating caption")
    duration_ms = (time.time() - t0) * 1000

    return {"caption": text, "processing_time_ms": round(duration_ms, 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
