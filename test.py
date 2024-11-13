import cv2
from ultralytics import YOLO


def process_video(input_video_path, output_video_path, model_path):
    # Load the pre-trained model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on frame
            results = model(
                frame, conf=0.25
            )  # You can adjust confidence threshold

            # Render detection results on frame
            annotated_frame = results[0].plot()

            # Write frame to output video
            out.write(annotated_frame)

            # Display frame (optional - comment out if running headless)
            cv2.imshow("Processing Video", annotated_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(
            f"Video processing completed. Output saved to: {output_video_path}"
        )


# Example usage
if __name__ == "__main__":
    input_video = (
        "./datasets/trimmed_video.mp4"  # Replace with your input video path
    )
    output_video = "output_detected.mp4"
    model_path = "./yolov8n.pt"  # Your trained model path

    process_video(input_video, output_video, model_path)
