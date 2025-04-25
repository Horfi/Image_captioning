# test_caption.py

import os
import random
from PIL import Image
from models.caption_model import CaptionModel
from utils.preprocess import preprocess_image

def main():
    # 1) Load the model
    model = CaptionModel()
    model.load_model("models/caption_model_500.keras")  # adjust path if yours differs

    # 2) Point to your Flickr8k (or other) images folder
    images_dir = os.path.join(os.path.dirname(__file__), "..", "data", "Flickr8k_Dataset")
    if not os.path.isdir(images_dir):
        raise RuntimeError(f"Images folder not found: {images_dir}")

    # 3) Pick a random image
    all_imgs = [f for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
    img_name = random.choice(all_imgs)
    img_path = os.path.join(images_dir, img_name)
    print(f"Selected image: {img_name}")

    # 4) Load & preprocess
    img = Image.open(img_path).convert("RGB")
    img_arr = preprocess_image(img)  # should give you the same array used in FastAPI

    # 5) Generate caption
    caption = model.generate_caption(img_arr)
    print("Generated caption:")
    print("   ", caption)

if __name__ == "__main__":
    main()
