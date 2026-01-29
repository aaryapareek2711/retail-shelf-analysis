"""
Training script for SKU-110K dataset.

NOTE:
Due to time and compute constraints, this script demonstrates
how the model would be fine-tuned on SKU-110K.
"""

from ultralytics import YOLO

def train_on_sku110k():
    # Load pretrained YOLO model
    model = YOLO("yolov8s.pt")

    # Path to SKU-110K dataset config (to be added)
    data_yaml = "sku110k.yaml"

    # Train / fine-tune model
    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=8
    )

if __name__ == "__main__":
    train_on_sku110k()
