from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

model = tf.keras.models.load_model("human_fire.keras")
classes = ["flame", "person", "unknown"]

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((96, 96))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    index = int(np.argmax(pred))

    return JSONResponse({
        "class": classes[index],
        "confidence": float(pred[index]),
        "scores": {
            classes[i]: float(pred[i])
            for i in range(len(classes))
        }
    })