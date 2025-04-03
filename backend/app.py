from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from models.caption_model import CaptionModel
from utils.preprocess import preprocess_image
import time
import uvicorn

app = FastAPI(title="Image Caption Generator API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
caption_model = CaptionModel()
caption_model.load_model("models/model_weights.h5")

@app.get("/")
def read_root():
    return {"message": "Image Caption Generator API is running"}

@app.post("/api/caption")
async def generate_caption(file: UploadFile = File(...)):
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess the image
        image_content = await file.read()
        img = Image.open(io.BytesIO(image_content))
        processed_image = preprocess_image(img)
        
        # Generate caption
        start_time = time.time()
        caption = caption_model.generate_caption(processed_image)
        processing_time = time.time() - start_time
        
        return {
            "caption": caption,
            "processing_time_ms": round(processing_time * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)