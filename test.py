from ultralytics import YOLO

base = r"F:\MY PROJECTS\VIDEO"
model = YOLO(fr"{base}\best_v8l_896.pt")

# Run on the trimmed clip; a rendered video will be saved under runs/detect/predict*/ in your cwd
results = model.predict(
    source=fr"{base}\signal_1min.mp4",
    save=True,        # save annotated video
    conf=0.25,        # adjust if you want stricter/looser detections
    iou=0.45,
    device=0          # use 0 for first GPU, or "cpu" to force CPU
)
