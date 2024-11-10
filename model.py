import torch
from ultralytics import YOLO


def train_model(config):
    """
    Train YOLO model with essential optimizations for M3 Pro
    """
    model = YOLO(config["model_path"])

    for dataset_path in config["datasets"]:
        print(f"\nTraining on dataset: {dataset_path}")

        try:
            result = model.train(
                data=dataset_path,
                epochs=config["epochs"],
                imgsz=config["image_size"],
                batch=config["batch_size"],
                device="mps",
                workers=config["num_workers"],
                patience=config["early_stopping_patience"],
                val=True,
                max_det=300,  # Prevent NMS timeout
                cache=True,  # Speed up data loading
            )

        except Exception as e:
            print(f"Error in training: {str(e)}")
            continue

    model.save(f"{config['project_name']}/final_trained_model.pt")
    print(f"\nTraining completed. Model saved.")

    return model


# Simple configuration optimized for M3 Pro
config = {
    "model_path": "yolov8n.pt",
    "datasets": [
        "./dataset/vehicle/data.yaml",
        "./dataset/traffic/data.yaml",
        "./dataset/people/data.yaml",
    ],
    "project_name": "yolo_training",
    "epochs": 20,
    "image_size": 640,
    "batch_size": 8,  # Balanced for M3 Pro memory
    "num_workers": 6,  # Good balance for 11 CPU cores
    "early_stopping_patience": 5,
}

if __name__ == "__main__":
    trained_model = train_model(config)
