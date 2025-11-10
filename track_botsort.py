# track_botsort.py  (clean copy)
import os
import csv
import cv2
import ultralytics
from ultralytics import YOLO

def main():

    BASE   = r"F:\MY PROJECTS\VIDEO"
    MODEL  = os.path.join(BASE, "best_v8l_896.pt")
    VIDEO  = os.path.join(BASE, "signal.mp4")    
    OUTCSV = os.path.join(BASE, "traj_botsort.csv")


    trk_dir = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers")
    botsort_yaml   = os.path.join(trk_dir, "botsort.yaml")
    bytetrack_yaml = os.path.join(trk_dir, "bytetrack.yaml")
    TRACKER_YAML = botsort_yaml if os.path.exists(botsort_yaml) else bytetrack_yaml
    print(f"[INFO] Using tracker: {os.path.basename(TRACKER_YAML)}")


    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"Cannot open video: {VIDEO}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"[INFO] FPS: {fps}")

   
    m = YOLO(MODEL)


    WANT = {"person","bicycle","rickshaw","cng","car","van","bus","truck"}
    MODEL_NAMES = set(m.names.values())
    KEEP = WANT & MODEL_NAMES
    if not KEEP:
        KEEP = MODEL_NAMES  # fallback
    print(f"[INFO] Keeping classes: {sorted(KEEP)}")


    stream = m.track(
        source=VIDEO,
        imgsz=896, conf=0.25, iou=0.5,
        tracker=TRACKER_YAML,
        persist=True, save=True, name='signal_botsort',
        stream=True
    )

   
    with open(OUTCSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","cls","frame","t","x","y","vx","vy","speed"])
        last = {}   
        frame_i = 0

        for r in stream:
            frame_i += 1
            t = frame_i / fps

            if r.boxes is None or r.boxes.id is None:
                continue

            ids  = r.boxes.id.int().cpu().tolist()
            clsi = r.boxes.cls.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()

            for tid, cidx, bb in zip(ids, clsi, xyxy):
                cname = m.names[int(cidx)]
                if cname not in KEEP:
                    continue

                x1, y1, x2, y2 = bb
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                if tid in last:
                    px, py, pt = last[tid]
                    dt = max(t - pt, 1e-6)
                    vx, vy = (cx - px) / dt, (cy - py) / dt
                else:
                    vx = vy = 0.0

                speed = (vx * vx + vy * vy) ** 0.5
                w.writerow([
                    tid, cname, frame_i, f"{t:.3f}",
                    f"{cx:.3f}", f"{cy:.3f}",
                    f"{vx:.3f}", f"{vy:.3f}", f"{speed:.3f}"
                ])
                last[tid] = (cx, cy, t)

    print(f"[DONE] Trajectories -> {OUTCSV}")
    print("[DONE] Annotated video -> runs/track/signal_botsort/")

if __name__ == "__main__":
    main()
