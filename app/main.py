from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from app.model.model import PPEDetector

app = FastAPI()
detector = PPEDetector()

@app.post("/detect_frame/")
async def detect_frame(image: UploadFile = File(...)):
    """Accept a single frame and return detections."""
    image_bytes = await image.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    detections = detector.detect(frame)
    return detections  # Returns bounding boxes, scores, and class_ids
