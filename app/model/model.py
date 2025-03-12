import cv2
import numpy as np
import onnxruntime as ort
import os

class PPEDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            #Replace the model you want
            model_path = os.path.join(os.path.dirname(__file__), "model1.onnx")

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = 0.3
        self.nms_threshold = 0.5

    def preprocess(self, frame):
        """Prepares image for inference"""
        original_shape = frame.shape[:2]
        frame = cv2.resize(frame, (640, 640))
        frame = frame / 255.0
        frame = frame.astype(np.float32)
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        return frame, original_shape

    def detect(self, frame):
        """Runs inference and processes results"""
        input_tensor, original_shape = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]

        boxes, scores, class_ids = self.postprocess(outputs, original_shape)
        return {"boxes": boxes, "scores": scores, "class_ids": class_ids}

    def postprocess(self, outputs, original_shape):
        """Processes model outputs and applies Non-Maximum Suppression (NMS)"""
        img_h, img_w = original_shape[:2]
        boxes, scores, class_ids = [], [], []

        outputs = np.transpose(outputs, (0, 2, 1))  # Reshape (1, 12, 8400) -> (1, 8400, 12)

        for det in outputs[0]:
            cx, cy, w, h = det[:4]

            # Scale bbox to original image
            cx = cx * img_w / 640
            cy = cy * img_h / 640
            w = w * img_w / 640
            h = h * img_h / 640

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

            conf = max(det[4:])
            class_id = np.argmax(det[4:])

            if conf > self.conf_threshold:
                boxes.append([x1, y1, x2, y2])
                scores.append(float(conf))  # Ensure scores are JSON-serializable
                class_ids.append(int(class_id))

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            scores = [scores[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]
        print(f'Boxes :{boxes}')
        print(f'Scores :{scores}')
        print(f'Class IDs :{class_ids}')
        return boxes, scores, class_ids
