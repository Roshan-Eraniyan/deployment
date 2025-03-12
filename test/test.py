import cv2
import requests
import numpy as np
import os
from collections import deque
from datetime import datetime

# API Endpoint
API_URL = "http://127.0.0.1:8000/detect_frame/"

# Video Paths
VIDEO_PATH = "inputs/input.mp4"
VIOLATIONS_DIR = "Violations"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Violation tracking
frame_buffer = deque(maxlen=int(fps * 2.5))  # Store 2.5 sec of frames
violation_clip = None
violation_active = False
violation_start_time = None

# Class names and violations
#Classes of model1
CLASS_NAMES = ["Boots", "Goggles", "Hardhat", "No_Boots", "No_Goggles", "No_Hardhat", "No_Safety_Vest", "Safety_Vest"]
VIOLATION_CLASSES = {"No_Boots", "No_Goggles", "No_Hardhat", "No_Safety_Vest"}  # Only these will be drawn

#Classes of model2, uncomment it you want to try model2
#CLASS_NAMES = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone",
#    "Safety Vest", "machinery", "vehicle"]
#VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}  # Only these will be drawn

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Add frame to buffer
    frame_buffer.append(frame.copy())

    # Send frame to API
    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(API_URL, files={"image": img_encoded.tobytes()})

    if response.status_code == 200:
        detections = response.json()
        boxes, scores, class_ids = detections["boxes"], detections["scores"], detections["class_ids"]

        has_violation = False
        for i, class_id in enumerate(class_ids):
            class_name = CLASS_NAMES[class_id]

            # Draw bounding box **ONLY** for violation classes
            if class_name in VIOLATION_CLASSES:
                has_violation = True
                x1, y1, x2, y2 = boxes[i]
                confidence = scores[i]

                # Draw violation in red
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # If a violation is detected, start recording
        if has_violation:
            if not violation_active:
                violation_active = True
                violation_start_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps  # Get start time in seconds

                # Create violation video path
                violation_video_path = os.path.join(VIOLATIONS_DIR,
                                                    f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                violation_clip = cv2.VideoWriter(violation_video_path, fourcc, fps, (frame_width, frame_height))

                # Write past frames to ensure pre-violation context
                for past_frame in frame_buffer:
                    violation_clip.write(past_frame)

        # If recording is active, write the current frame
        if violation_active:
            violation_clip.write(frame)

            # Stop recording after 5 seconds
            current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            if current_time - violation_start_time >= 5:
                violation_clip.release()
                violation_active = False
                violation_clip = None  # Reset video writer


# Release resources
cap.release()
cv2.destroyAllWindows()
if violation_clip:
    violation_clip.release()

print("Processing completed. Check 'outputs/' for videos.")
