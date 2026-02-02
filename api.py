from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Retail Shelf Detection API")

# Load YOLO model once
model = YOLO("yolov8s.pt")


# Root endpoint
@app.get("/")
def root():
    return {"message": "Retail Shelf Detection API is running"}


# Prediction endpoint
@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run YOLO with tuned thresholds (IMPORTANT)
    results = model.predict(
        pil_image,
        conf=0.4,     # higher confidence = fewer false boxes
        iou=0.5,      #remove overlapping boxes
        imgsz=640
    )

    # Convert image to NumPy (OpenCV format)
    annotated = np.array(pil_image)

    # Draw ONLY GREEN boxes
    for box in results[0].boxes:
        confidence = float(box.conf[0])

        # Extra safety filter
        if confidence < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = (0, 255, 0)  # GREEN (BGR)
        thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        cv2.putText(
            annotated,
            f"{confidence:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # Convert back to image bytes
    annotated_pil = Image.fromarray(annotated)
    buffer = io.BytesIO()
    annotated_pil.save(buffer, format="JPEG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg"
    )


