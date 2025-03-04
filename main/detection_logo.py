import torch
import cv2
import numpy as np

# Load YOLOv5 Model
model_path = r"E:\Stream_Censor\DeepErase-X\models\openlogo.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

# Set Model Parameters
model.conf = 0.60  # Confidence threshold
model.iou = 0.45   # NMS IOU threshold
model.classes = None  # Detect all classes
model.eval()

def detect_logo_in_frame(frame):
    """
    Detects logos in a given frame using the YOLOv5 model.

    Args:
        frame (numpy.ndarray): The input image/frame (BGR format).

    Returns:
        tuple:
            - List of bounding boxes [(xmin, ymin, xmax, ymax, class_name, confidence), ...]
            - Processed frame with detections drawn
    """
    logo_frame = frame.copy()
    # Convert BGR to RGB for YOLOv5 processing
    rgb_frame = cv2.cvtColor(logo_frame, cv2.COLOR_BGR2RGB)

    # Perform Object Detection
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]  # Get detections as a pandas DataFrame

    bbox_list = []  # List to store bounding boxes

    # Draw Bounding Boxes and Store Detections
    for _, detection in detections.iterrows():
        class_name = detection["name"]
        confidence = detection["confidence"]
        xmin, ymin, xmax, ymax = int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"]), int(detection["ymax"])

        # Store bbox as (xmin, ymin, xmax, ymax, class_name, confidence)
        bbox_list.append((xmin, ymin, xmax, ymax))

        # Draw Rectangle and Label
        cv2.rectangle(logo_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(logo_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return bbox_list, logo_frame  # Return detected bounding boxes and processed frame
