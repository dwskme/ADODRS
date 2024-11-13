from pathlib import Path

import torch
from ultralytics import YOLO

# Basic environment check
print(
    f"PyTorch device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}"
)

# Define paths
DATASET_PATH = Path("./datasets/coco_splitted/")
RESULTS_PATH = Path("./results")
RESULTS_PATH.mkdir(exist_ok=True)

# Simplified training parameters
params = {
    "data_yaml": str(DATASET_PATH / "data.yaml"),
    "img_size": 320,  # Small image size for faster training
    "batch_size": 8,  # Increased batch size for faster training
    "epochs": 10,  # Reduced epochs
    "patience": 5,  # Early stopping if no improvement
    "model_type": "yolov8n.pt",  # Using nano model for faster training
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "lr0": 0.01,
    "lrf": 0.01,
    "resume": True,  # Enable training resume if stopped
    "exist_ok": True,  # Continue training from last checkpoint
    "save_period": 1,  # Save after every epoch to allow resume
    "project": str(RESULTS_PATH),
    "name": "train",
}


def train_model(params):
    """Train YOLOv8 model with simplified parameters"""
    # Load model
    model = YOLO(params["model_type"])

    # Start training
    results = model.train(
        data=params["data_yaml"],
        imgsz=params["img_size"],
        epochs=params["epochs"],
        batch=params["batch_size"],
        save_period=params["save_period"],  # Save checkpoints for resumability
        device=params["device"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        patience=params["patience"],
        resume=params["resume"],
        exist_ok=params["exist_ok"],
        project=params["project"],
        name=params["name"],
    )

    return model, results


if __name__ == "__main__":
    model, results = train_model(params)
