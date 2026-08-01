from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

# Load model on startup
model = tf.keras.models.load_model("human_fire.keras")
classes = ["flame", "person", "unknown"]

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
async def predict(request: Request):
    # 1. Read the raw binary JPEG bytes sent directly by the ESP32
    image_bytes = await request.body()
    
    if not image_bytes:
        return JSONResponse(status_code=400, content={"error": "No image provided"})

    try:
        # 2. Open the image directly from the byte stream
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((96, 96))

        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        # 3. Run Inference
        pred = model.predict(img, verbose=0)[0]

        # Map predictions to classes
        scores = {classes[i]: float(pred[i]) for i in range(len(classes))}

        # 4. Return the EXACT flat JSON structure the ESP32 expects
        return JSONResponse({
            "flame": scores.get("flame", 0.0),
            "person": scores.get("person", 0.0)
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to process image"})