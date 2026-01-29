# Retail Shelf Analysis – Computer Vision System

## Overview
This project implements an end-to-end computer vision pipeline for retail shelf analysis. The system detects products on retail shelves, groups them by brand similarity, generates visualizations, and exposes results via an API.

## Dataset
The intended dataset for training is **SKU-110K**, a large-scale dense retail shelf dataset designed for product detection tasks. Due to time and compute constraints, a pretrained YOLOv8 model was used to validate the full inference and deployment pipeline. The project includes training code structured specifically for fine-tuning on SKU-110K.

## Model
- YOLOv8 (pretrained)
- Designed for fine-tuning on SKU-110K
- Handles dense object detection on retail shelves

## Training
A training script (`training/train.py`) is provided to demonstrate how the model can be fine-tuned on the SKU-110K dataset. The script uses pretrained weights and is configurable for epochs, batch size, and image resolution.

## Pipeline
Image → Product Detection → Brand Grouping → Visualization → JSON Response

## Backend
- FastAPI
- POST `/predict`
- Accepts image uploads
- Returns bounding boxes, confidence scores, group IDs, and visualization path

## Frontend
- Streamlit-based user interface
- Upload image
- Display detection results and JSON output

## Deployment
The project is structured for deployment on **Modal** with GPU support (A10G). A deployment-ready skeleton is provided in `modal_app.py` to demonstrate cloud deployment readiness.

## Notes
Due to resource and time constraints, full-scale training on SKU-110K and live GPU deployment were not executed. However, the system architecture and code are designed to support both seamlessly.
