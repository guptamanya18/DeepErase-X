import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from detection_nlp import detect_text_in_frame
from removal_file import create_mask, apply_inpainting
import json
from detection_logo import detect_logo_in_frame


def expand_bbox(bbox, img_width, img_height, expansion=10):
    """Expands a bounding box by a fixed number of pixels while keeping it within image bounds."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - expansion)
    y_min = max(0, y_min - expansion)
    x_max = min(img_width, x_max + expansion)
    y_max = min(img_height, y_max + expansion)
    return [x_min, y_min, x_max, y_max]


def merge_bounding_boxes(bboxes, threshold=20):
    """Merges nearby bounding boxes based on a proximity threshold."""
    if not bboxes:
        return []

    merged = []
    bboxes.sort()  # Sort based on x_min

    while bboxes:
        current = bboxes.pop(0)  # Take first bbox
        x_min, y_min, x_max, y_max = current

        i = 0
        while i < len(bboxes):
            other = bboxes[i]
            ox_min, oy_min, ox_max, oy_max = other

            # Check if the boxes are close enough to merge
            if (abs(x_min - ox_max) < threshold or abs(x_max - ox_min) < threshold) and \
               (abs(y_min - oy_max) < threshold or abs(y_max - oy_min) < threshold):

                # Merge the boxes by taking min/max coordinates
                x_min, y_min = min(x_min, ox_min), min(y_min, oy_min)
                x_max, y_max = max(x_max, ox_max), max(y_max, oy_max)
                bboxes.pop(i)  # Remove merged box
            else:
                i += 1

        merged.append([x_min, y_min, x_max, y_max])

    return merged


def overlay_masks_on_image(image, pred_masks, alpha=0.5):
    """Overlays instance segmentation masks on the original image."""
    if pred_masks is None or len(pred_masks) == 0:
        print("❌ No segmentation masks available!")
        return image

    # Convert to NumPy if needed
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()

    # Generate random colors for each mask
    num_masks = pred_masks.shape[0]
    mask_colors = np.random.randint(0, 255, (num_masks, 3), dtype=np.uint8)

    # Create an overlay for masks
    mask_overlay = np.zeros_like(image, dtype=np.uint8)

    for i in range(num_masks):
        mask = pred_masks[i]  # Get binary mask for instance i
        color = mask_colors[i]  # Assign random color
        mask_overlay[mask] = color  # Apply color to mask

    # Blend mask overlay with the original image
    blended = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
    return blended


def process_image(image_path, output_folder):
    """Processes an image: detects text, creates a merged mask, applies inpainting, and saves results."""

    # ✅ Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # ✅ Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image {image_path}")
        return

    img_height, img_width = image.shape[:2]  # Get image dimensions
    print("🔍 Processing Image...")

    # ✅ Detect text & extract segmentation masks
    results = detect_text_in_frame(image)
    pred_masks = results.get("pred_masks")  # Extract predicted segmentation masks

    bboxes, processed_frame = detect_logo_in_frame(image)
    logo_detection_path = os.path.join(output_folder, f"logo_detected.jpg")
    cv2.imwrite(logo_detection_path, processed_frame)

    # ✅ Extract & Convert Bounding Boxes to Integers (Ignoring Class 0)
    bounding_boxes = []
    for detection in results.get("detections", []):
        if "bbox" in detection and "class_id" in detection:
            if detection["class_id"] != 0:  # Ignore text (class 0)
                bbox = detection["bbox"]
                int_bbox = [int(coord) for coord in bbox]  # Convert float → int
                expanded_bbox = expand_bbox(int_bbox, img_width, img_height, expansion=10)  # Expand
                bounding_boxes.append(expanded_bbox)

    # ✅ Merge nearby bounding boxes
    merged_bounding_boxes = merge_bounding_boxes(bounding_boxes, threshold=20)

    # ✅ Create and Save Bounding Box Mask
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    bbox_mask = create_mask(img_pil, merged_bounding_boxes)
    bbox_mask_output_path = os.path.join(output_folder, "bbox_mask.png")
    cv2.imwrite(bbox_mask_output_path, bbox_mask)

    # ✅ Overlay segmentation masks on the image
    image_with_masks = overlay_masks_on_image(image, pred_masks)

    # ✅ Combine Bounding Box Mask & Predicted Segmentation Mask
    final_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Create empty mask
    if pred_masks is not None:
        for mask in pred_masks:
            final_mask[mask] = 255  # Add segmentation mask areas

    # Apply bounding box mask as well
    for bbox in merged_bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        final_mask[y_min:y_max, x_min:x_max] = 255  # Fill bounding box area

    for xmin, ymin, xmax, ymax in bboxes:
        final_mask[ymin:ymax, xmin:xmax] = 255  # Apply mask over logos

    # ✅ Save the final combined mask
    final_mask_output_path = os.path.join(output_folder, "final_mask.png")
    cv2.imwrite(final_mask_output_path, final_mask)

    # ✅ Apply Inpainting with Final Mask
    inpainted_image = apply_inpainting(image, final_mask, output_folder)
    inpaint_output_path = os.path.join(output_folder, "image_inpainted.png")
    cv2.imwrite(inpaint_output_path, inpainted_image)

    # ✅ Display the final mask
    plt.figure(figsize=(10, 6))
    plt.imshow(final_mask, cmap="gray")
    plt.axis("off")
    plt.title("Final Combined Mask")
    plt.show()

    # ✅ Display the inpainted image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Inpainted Image")
    plt.show()

    print(f"✅ Processed image saved at: {image_path}")
    print(f"✅ Mask saved at: {bbox_mask_output_path}")
    print(f"✅ Final Mask saved at: {final_mask_output_path}")
    print(f"✅ Inpainted image saved at: {inpaint_output_path}")
    print(f"✅ Bounding Boxes (Merged, Int Format): {merged_bounding_boxes}")
    print("🎉 Image processing complete!")


def process_video(video_path, output_folder, output_video_path):
    """Processes a video frame-by-frame, applies text/object detection, masking, and inpainting."""

    # ✅ Ensure output directories exist
    frames_folder = os.path.join(output_folder, "segement_frames")
    json_folder = os.path.join(output_folder, "json")
    inpainted_frames_folder = os.path.join(output_folder, "inpainted_frames")
    logo_detection_folder = os.path.join(output_folder, "detected_frames")
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(inpainted_frames_folder, exist_ok=True)
    os.makedirs(logo_detection_folder, exist_ok=True)

    # ✅ Open Video File
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        return

    # ✅ Get Video Properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        frame_count += 1
        print(f"🔍 Processing Frame {frame_count}...")
        img_height, img_width = frame.shape[:2]
        # ✅ Detect text and objects in frame
        results = detect_text_in_frame(frame)

        pred_masks = results.get("pred_masks")  # Extract predicted segmentation masks

        bboxes, processed_frame = detect_logo_in_frame(frame)
        logo_detection_path = os.path.join(logo_detection_folder, f"logo_detected_{frame_count}.jpg")
        cv2.imwrite(logo_detection_path, processed_frame)

        # ✅ Extract & Convert Bounding Boxes to Integers (Ignoring Class 0)
        bounding_boxes = []
        for detection in results.get("detections", []):
            if "bbox" in detection and "class_id" in detection:
                if detection["class_id"] != 0:  # Ignore text (class 0)
                    bbox = detection["bbox"]
                    int_bbox = [int(coord) for coord in bbox]  # Convert float → int
                    expanded_bbox = expand_bbox(int_bbox, img_width, img_height, expansion=10)  # Expand
                    bounding_boxes.append(expanded_bbox)

        # ✅ Merge nearby bounding boxes
        merged_bounding_boxes = merge_bounding_boxes(bounding_boxes, threshold=20)

        # ✅ Create and Save Bounding Box Mask
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        bbox_mask = create_mask(img_pil, merged_bounding_boxes)
        bbox_mask_output_path = os.path.join(output_folder, "bbox_mask.png")
        cv2.imwrite(bbox_mask_output_path, bbox_mask)

        # ✅ Overlay segmentation masks on the image
        image_with_masks = overlay_masks_on_image(frame, pred_masks)

        # ✅ Combine Bounding Box Mask & Predicted Segmentation Mask
        final_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)  # Create empty mask
        if pred_masks is not None:
            for mask in pred_masks:
                final_mask[mask] = 255  # Add segmentation mask areas

        # Apply bounding box mask as well
        for bbox in merged_bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            final_mask[y_min:y_max, x_min:x_max] = 255  # Fill bounding box area

        # ✅ Mask logo bounding boxes from `detect_logo_in_frame()`
        for xmin, ymin, xmax, ymax in bboxes:
            final_mask[ymin:ymax, xmin:xmax] = 255  # Apply mask over logos

        # ✅ Apply Inpainting
        inpainted_frame = apply_inpainting(frame, final_mask, output_folder)
        inpainted_frame_path = os.path.join(inpainted_frames_folder, f"inpainted_{frame_count}.jpg")
        cv2.imwrite(inpainted_frame_path, inpainted_frame)

        # ✅ Save JSON results
        json_path = os.path.join(json_folder, f"frame_{frame_count}.json")
        with open(json_path, "w") as f:
            json.dump({"detections": results["detections"], "filtered_bounding_boxes": bounding_boxes}, f, indent=4)

        # ✅ Save detected frame
        detected_frame_path = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(detected_frame_path, results["visualized_image"])
        # ✅ Write Inpainted Frame to Video
        out.write(inpainted_frame)

    # ✅ Release Video Capture & Writer
    cap.release()
    out.release()

    print("\n🎉 Video processing complete! Inpainted video saved at:", output_video_path)


# Example Usage:
process_video(r"E:\Stream_Censor\DeepErase-X\assets\output_video.mp4", r"E:\Stream_Censor\DeepErase-X\testing_output",
              r"/DeepErase-X/assets/testing_output\final_video.mp4")
# process_image(r"E:\Stream_Censor\DeepErase-X\testing_data\example_img.jpg", r"E:\Stream_Censor\DeepErase-X\testing_output")
