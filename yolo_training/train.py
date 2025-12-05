from ultralytics import YOLO
import numpy as np
import torch

model = YOLO("yolo11s.pt")
model.train(data="Food_Detector.v1-first_test.yolov11/data.yaml", epochs=100, imgsz=640, device = "cuda")
results = model.val()
# 위에까지 하면 파일 어딘가에 best.pt 나올 것. 그 다음에 그걸 불러와서 하기... 또는 모델을 export 해서 불러올 방법 찾아보기.
success = model.export()
