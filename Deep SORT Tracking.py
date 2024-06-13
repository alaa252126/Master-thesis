"""""
The code below until the END statement is taken from the source below:
Title : YOLOv8 -Deep SORT 
Aruthor : Ultralytics  , Nicolai Wojke
Date :2024
folder : ultralytics/trackers,and deep_sort
 Availability:https://github.com/ultralytics/ultralytics ,https://github.com/nwojke/deep_sort

 
with using AI tools like ChatGPT also

"""



import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

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

    # Initialize Deep SORT tracker
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, stream=True)

            for result in results:  # Iterate over the generator
                if result is not None and hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xywh.cpu().numpy() if result.boxes.xywh is not None else []
                    confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
                    class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

                    # Convert YOLO detections to the format required by Deep SORT
                    detections = [((x, y, w, h), confidence, class_id) for (x, y, w, h), confidence, class_id in zip(boxes, confidences, class_ids)]
                    
                    # Update the tracker with the current frame detections
                    tracked_objects = tracker.update_tracks(detections, frame=frame)

                    annotated_frame = frame.copy()
                    for obj in tracked_objects:
                        if not obj.is_confirmed():
                            continue
                        track_id = obj.track_id
                        ltrb = obj.to_ltrb()  # left, top, right, bottom
                        cv2.rectangle(annotated_frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        
                        # Update track history for polyline drawing
                        x_center, y_center = (ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2
                        track_history[track_id].append((x_center, y_center))
                        if len(track_history[track_id]) > 30:
                            track_history[track_id].pop(0)
                        points = np.array(track_history[track_id]).reshape((-1, 1, 2)).astype(np.int32)
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
