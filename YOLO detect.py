"""""
The code below until the END statement is taken from the source below:
Title : YOLOv8
Aruthor : Ultralytics YOLOv8
Date :2024
folder : ultralytics/models/yolo/detect
 Availability:https://github.com/ultralytics/ultralytics


"""



import os
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
#model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data= "fjord.yaml",batch = -1, epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 
results = model.predict("/home/student/alaaabo/data/images/test",save=True, conf=0.5,iou=0.7)  # predict on an test set
#path = model.export(format="onnx")  # export the model to ONNX format
#file_path = r"C:\Users\alaa abo\OneDrive - Universitetet i Stavanger\YOLO\config.yaml"
#print("File path:", file_path)  # Debugging output
#model.train(data=file_path, epochs=1)  # train the model



