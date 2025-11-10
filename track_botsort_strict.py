# track_botsort_strict.py
import os, csv, cv2, ultralytics
import numpy as np
from ultralytics import YOLO

BASE   = r"F:\MY PROJECTS\VIDEO"
MODEL  = os.path.join(BASE, "best_v8l_896.pt")
VIDEO  = os.path.join(BASE, "signal.mp4")    
OUTCSV = os.path.join(BASE, "traj_botsort_raw.csv") 
RUNNAME= "signal_botsort_strict"


DET_CONF = 0.50        
MIN_AREA = 32*32       
IMG_SIZE = 896
IOU      = 0.5

# prefer BoT-SORT; fallback to ByteTrack
trk_dir = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers")
botsort_yaml   = os.path.join(trk_dir, "botsort.yaml")
bytetrack_yaml = os.path.join(trk_dir, "bytetrack.yaml")
TRACKER = botsort_yaml if os.path.exists(botsort_yaml) else bytetrack_yaml
print("[INFO] Tracker:", os.path.basename(TRACKER))



cap = cv2.VideoCapture(VIDEO); assert cap.isOpened(), f"Cannot open {VIDEO}"
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"[INFO] Video fps={fps:.3f} size={W}x{H}")

m = YOLO(MODEL)
KEEP = {"person","bicycle","rickshaw","cng","car","van","bus","truck"} & set(m.names.values())

stream = m.track(
    source=VIDEO,
    imgsz=IMG_SIZE, conf=max(DET_CONF-0.1, 0.25), iou=IOU,   
    tracker=TRACKER, persist=True, save=True, name=RUNNAME,
    stream=True
)

with open(OUTCSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "id","cls","conf","frame","t",
        "x","y","vx","vy","speed",
        "x1","y1","x2","y2","w","h","area"
    ])
    last = {}  
    frame_i = 0
    for r in stream:
        frame_i += 1
        t = frame_i / fps

        if not (r.boxes is not None and r.boxes.id is not None):
            continue

        ids  = r.boxes.id.int().cpu().tolist()
        clsi = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.zeros(len(ids))
        xyxy = r.boxes.xyxy.cpu().numpy()

        for tid, cidx, cf, bb in zip(ids, clsi, conf, xyxy):
            cname = m.names[int(cidx)]
            if cname not in KEEP:
                continue

            x1,y1,x2,y2 = bb
            w_px = max(0.0, x2-x1); h_px = max(0.0, y2-y1); area = w_px*h_px
            if cf < DET_CONF or area < MIN_AREA:
                continue  

            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            if tid in last:
                px, py, pt = last[tid]
                dt = max(t-pt, 1e-6)
                vx, vy = (cx-px)/dt, (cy-py)/dt
            else:
                vx = vy = 0.0

            speed = (vx*vx + vy*vy)**0.5
            w.writerow([
                int(tid), cname, f"{cf:.3f}", frame_i, f"{t:.3f}",
                f"{cx:.3f}", f"{cy:.3f}", f"{vx:.3f}", f"{vy:.3f}", f"{speed:.3f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{w_px:.1f}", f"{h_px:.1f}", f"{area:.0f}"
            ])
            last[tid] = (cx,cy,t)

print("[DONE] strict trajectories ->", OUTCSV)
print("[DONE] annotated video    -> runs/track/%s" % RUNNAME)
