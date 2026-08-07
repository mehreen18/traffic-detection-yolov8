import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

MODEL_PATH = "/content/drive/MyDrive/traffic_app/models/best.pt"
CONF_THRESHOLD = 0.35

app = FastAPI(title="Traffic Object Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Traffic Detection API running. POST an image to /predict."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(source=np.array(image), conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
        })

    annotated = result.plot()
    annotated_img = Image.fromarray(annotated[:, :, ::-1])
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image_base64": annotated_b64,
    })
