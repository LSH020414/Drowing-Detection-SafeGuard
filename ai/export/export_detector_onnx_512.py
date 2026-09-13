from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

model = YOLO(str(MODEL_PATH))

result = model.export(
    format="onnx",
    imgsz=512,
    dynamic=False,
    simplify=True,
    opset=13,
)

print("EXPORT RESULT:", result)