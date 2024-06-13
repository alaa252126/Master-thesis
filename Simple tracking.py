"""""
The code below until the END statement is taken from the source below:
Title : YOLOv8
Aruthor : Ultralytics YOLOv8
Date :2024
folder : ultralytics/trackers
 Availability:https://github.com/ultralytics/ultralytics
with using AI tools like ChatGPT 


"""


import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    logger.info("Loading model")
    return YOLO(model_path)

def train_model(model, data_path, batch_size, epochs):
    logger.info("Starting training")
    model.train(data=data_path, batch=batch_size, epochs=epochs)
    logger.info("Training completed")

def evaluate_model(model):
    logger.info("Evaluating model")
    metrics = model.val()
    logger.info("Model evaluation completed")
    return metrics

def predict_model(model, test_images_path, save, conf, iou):
    logger.info("Starting prediction")
    results = model.predict(test_images_path, save=save, conf=conf, iou=iou)
    logger.info("Prediction completed")
    return results

def process_video(model, video_path, output_path):
    VIDEO_DIR = "/home/student/alaaabo/data/images"
    video_path = os.path.join(VIDEO_DIR, "202405300240.mp4")
    output_dir = "/home/student/alaaabo/data/runs/detect/traker"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    video_path_out = os.path.join(output_dir, "202405300240.mp4_out.mp4")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    track_history = defaultdict(list)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, stream=True)

            for result in results:  # Iterate over the generator
                if result is not None and hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xywh.cpu().numpy() if result.boxes.xywh is not None else []
                    track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

                    annotated_frame = result.plot()
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 30 tracks for 30 frames
                            track.pop(0)

                        points = np.array(track).reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                    out.write(annotated_frame)

            frame_count += 1
            
        else:
            break

    cap.release()
    out.release()

if __name__ == "__main__":
    MODEL_PATH = "yolov8m.pt"
    DATA_PATH = "fjord.yaml"
    TEST_IMAGES_PATH = "/home/student/alaaabo/data/images/test"
    VIDEO_PATH = "/home/student/alaaabo/data/images/202405300240.mp4"
    OUTPUT_VIDEO_PATH = "/home/student/alaaabo/data/runs/detect/traker/202405300240.mp4_out.mp4"
    BATCH_SIZE = -1
    EPOCHS = 1
    CONFIDENCE = 0.5
    IOU_THRESHOLD = 0.7

    model = load_model(MODEL_PATH)
    train_model(model, DATA_PATH, BATCH_SIZE, EPOCHS)
    evaluate_model(model)
    predict_model(model, TEST_IMAGES_PATH, save=True, conf=CONFIDENCE, iou=IOU_THRESHOLD)
    process_video(model, VIDEO_PATH, OUTPUT_VIDEO_PATH)
