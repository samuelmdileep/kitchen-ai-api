from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

model = tf.keras.models.load_model("human_fire.keras")
classes = ["flame", "person", "unknown"]

# 👇 Global variable to hold the latest image in memory
latest_image_bytes = None

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
async def predict(request: Request):
    global latest_image_bytes
    
    # 1. Read the bytes
    image_bytes = await request.body()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"error": "No image provided"})

    # 👇 2. Save the image to memory so the dashboard can access it
    latest_image_bytes = image_bytes

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((96, 96))
        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0]
        scores = {classes[i]: float(pred[i]) for i in range(len(classes))}

        return JSONResponse({
            "flame": scores.get("flame", 0.0),
            "person": scores.get("person", 0.0)
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to process image"})

# 👇 NEW ENDPOINT: Serve the image to your web browser / dashboard
@app.get("/latest-image")
def get_latest_image():
    global latest_image_bytes
    if latest_image_bytes is None:
        return JSONResponse(status_code=404, content={"error": "No image captured yet"})
    
    # Return the raw JPEG exactly as a browser expects it
    return Response(content=latest_image_bytes, media_type="image/jpeg")