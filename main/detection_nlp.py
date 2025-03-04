import cv2
import torch
import json
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

# Setup logger
setup_logger()

# ✅ Load Detectron2 config
cfg = get_cfg()
cfg.merge_from_file(r"E:\Stream_Censor\DeepErase-X\assets\github\TextFuseNet\configs\ocr\icdar2015_101_FPN.yaml")
cfg.MODEL.WEIGHTS = r"E:\Stream_Censor\DeepErase-X\models\model_ic15_r101.pth"  # Ensure correct path
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15  # Lower threshold to detect faint text
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # Adjust NMS threshold to prevent duplicate boxes
cfg.INPUT.MIN_SIZE_TEST = 1200  # Increase input resolution for better detection
cfg.INPUT.MAX_SIZE_TEST = 2000  # Set a reasonable max size
cfg.INPUT.RANDOM_FLIP = "horizontal"  # Enable random flipping
cfg.INPUT.MIN_SIZE_TRAIN = (800, 1000, 1200)  # Multi-scale training
cfg.INPUT.CROP.ENABLED = True  # Enable cropping augmentation
cfg.INPUT.CROP.SIZE = [0.7, 0.7]  # Crop to 70% of the original image

cfg.freeze()

# ✅ Initialize Model
demo = VisualizationDemo(cfg)


def detect_text_in_frame(frame):
    """
    Detects text in the given frame using Detectron2.

    Args:
        frame (numpy array): Input image/frame in OpenCV format.

    Returns:
        dict: {
            "detections": List of text regions detected,
            "visualized_image": Image with bounding boxes drawn
        }
    """
    if frame is None:
        raise ValueError("❌ Error: Input frame is None!")

    # ✅ Run Detection
    predictions, vis_output, pred_masks = demo.run_on_image(frame)

    # ✅ Extract detected characters and bounding boxes
    instances = predictions["instances"]
    num_instances = len(instances)
    detections = []

    if num_instances == 0:
        print("⚠️ No text detected in the frame.")
    else:
        print(f"✅ {num_instances} text regions detected.")

        for i in range(num_instances):
            bbox = instances.pred_boxes[i].tensor.cpu().numpy().tolist()[0]  # Convert to list
            score = float(instances.scores[i].cpu().numpy())  # Confidence score
            class_id = int(instances.pred_classes[i].cpu().numpy())  # Class ID (text character)

            # Store data
            detections.append({
                "character_id": i + 1,
                "bbox": bbox,
                "confidence": score,
                "class_id": class_id
            })

    return {
        "detections": detections,
        "visualized_image": vis_output.get_image()[:, :, ::-1],  # Convert from RGB to BGR for OpenCV
        "pred_masks": pred_masks
    }