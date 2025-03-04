import cv2
import json
import os
from detection_nlp import detect_text_in_frame
from removal_file import create_mask
from removal_file import apply_inpainting


# ✅ Define Input Video & Output Folder
video_path = r"E:\Stream_Censor\DeepErase-X\assets\output_video.mp4"
output_folder = r"E:\Stream_Censor\DeepErase-X\testing_output"

# ✅ Ensure output directories exist
frames_folder = os.path.join(output_folder, "frames")
json_folder = os.path.join(output_folder, "json")

os.makedirs(frames_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)

# ✅ Open Video File
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video file!")
    exit(1)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    frame_count += 1
    print(f"🔍 Processing Frame {frame_count}...")

    # ✅ Detect text in frame
    results = detect_text_in_frame(frame)

    # ✅ Save JSON results
    json_path = os.path.join(json_folder, f"frame_{frame_count}.json")
    with open(json_path, "w") as f:
        json.dump(results["detections"], f, indent=4)

    print(f"✅ JSON saved: {json_path}")

    # ✅ Save detected frame
    frame_path = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, results["visualized_image"])
    print(f"✅ Frame saved: {frame_path}")

# ✅ Release Video Capture
cap.release()
cv2.destroyAllWindows()

print("\n🎉 Video processing complete! All frames and JSON files saved.")
