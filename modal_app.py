"""
Modal deployment script for Retail Shelf Analysis.

This script demonstrates how the FastAPI inference service
can be deployed on Modal with GPU support.
"""

import modal

stub = modal.Stub("retail-shelf-analysis")

image = modal.Image.debian_slim().pip_install(
    "ultralytics",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "torch",
    "opencv-python"
)

@stub.function(
    image=image,
    gpu="A10G",
    timeout=600
)
def run_inference():
    """
    Entry point for GPU-backed inference on Modal.
    """
    print("Retail Shelf Analysis service running on Modal GPU")
