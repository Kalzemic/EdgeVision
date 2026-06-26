from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # downloads pretrained weights on first run
results = model.train(
    data='data.yaml',
    epochs=60,
    imgsz=800,
    batch=16,
    device=0,        # GPU index, or 'cpu'
    project='runs/edge_detector',
    name='EdgeVision_800',
)

print(results.box.map)       # mAP@0.5:0.95
print(results.box.map50)     # mAP@0.5
print(results.box.mp)        # mean precision
print(results.box.mr)      
