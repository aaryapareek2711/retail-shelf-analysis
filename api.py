from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import uuid

app = FastAPI(title="Retail Shelf Detection API")

# Load YOLO model once at startup
model = YOLO("yolov8s.pt")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Create required folders
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save uploaded image
    image_id = str(uuid.uuid4())
    input_path = f"inputs/{image_id}.jpg"
    output_path = f"outputs/{image_id}.jpg"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Run YOLO detection
    results = model.predict(
        source=input_path,
        conf=0.05,
        imgsz=640,
        save=True
    )

    objects = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            objects.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(confidence, 2),
                "group_id": f"brand_{int(y1 // 100)}"
# placeholder for now
            })

    # Find latest YOLO output image
    detect_dir = "runs/detect"
    latest_run = sorted(os.listdir(detect_dir))[-1]
    predicted_image = os.path.join(detect_dir, latest_run, os.path.basename(input_path))

    if os.path.exists(predicted_image):
        shutil.copy(predicted_image, output_path)

    return {
        "objects": objects,
        "visualization_path": output_path
    }
