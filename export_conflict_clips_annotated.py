# Robust exporter: stable labels, skip ped-ped, tolerant to fps/length issues
import os, cv2, pandas as pd
from collections import Counter

BASE   = r"F:\MY PROJECTS\VIDEO"
# IMPORTANT: use the SAME source video used for tracking/CSV times
VIDEO = r"F:\MY PROJECTS\VIDEO\signal.mp4"    
TRAJ  = r"F:\MY PROJECTS\VIDEO\traj_botsort_clean.csv" 
CONF   = os.path.join(BASE, "conflicts_ttc_pet.csv")
OUTDIR = os.path.join(BASE, "conflict_clips_marked")

PRE_S, POST_S = 3.0, 3.0
MAX_N = 15
TRAIL = 12
KEEP_TYPES = {"veh-veh","veh-nmv","veh-ped"}  # ignore ped-ped

def detect_xy_cols(df):
    if {"x","y"}.issubset(df.columns): return "x","y"
    if {"x_px","y_px"}.issubset(df.columns): return "x_px","y_px"
    raise ValueError("Trajectory CSV must contain x,y or x_px,y_px.")

def stable_label(series):
    c = Counter([str(s) for s in series if pd.notna(s) and str(s)!=""])
    return c.most_common(1)[0][0] if c else ""

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df_tr = pd.read_csv(TRAJ)
    df_cf = pd.read_csv(CONF)
    if df_cf.empty:
        print("No conflicts to export."); return

    # filter types and drop ped-ped
    if "type" in df_cf.columns:
        df_cf = df_cf[df_cf["type"].isin(KEEP_TYPES)]
        if df_cf.empty:
            print("No conflicts after type filter."); return

    xcol, ycol = detect_xy_cols(df_tr)
    df_tr["id"] = df_tr["id"].astype(int)
    df_tr["frame"] = df_tr["frame"].astype(int)
    if "cls" not in df_tr.columns: df_tr["cls"] = ""

    # stable class per track
    stable_cls = df_tr.groupby("id")["cls"].apply(stable_label).to_dict()

    # trajectories indexed by (id, frame)
    tr_by_id = {int(tid): g[["frame", xcol, ycol]].drop_duplicates("frame")
                           .set_index("frame").sort_index()
                for tid, g in df_tr.groupby("id")}

    # top-N by severity
    df_cf["severity"] = df_cf[["ttc_s","pet_s"]].min(axis=1)
    top = df_cf.sort_values("severity").head(MAX_N).reset_index(drop=True)

    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"Cannot open {VIDEO}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Using video: {VIDEO}  fps={fps:.3f}  size={W}x{H}")

    saved, skipped = 0, 0
    for i, r in top.iterrows():
        id_a, id_b = int(r["id_a"]), int(r["id_b"])
        la, lb = stable_cls.get(id_a, ""), stable_cls.get(id_b, "")
        t_event = float(r["t"])

        # frame window (no clamping to total length)
        f0 = max(int(round((t_event - PRE_S) * fps)), 0)
        f1 = max(int(round((t_event + POST_S) * fps)), f0 + 1)

        # set up writer
        out_name = f"ev_{i:02d}_{r['type']}_t{t_event:.2f}_ttc{r['ttc_s']:.2f}_pet{r['pet_s']:.2f}.mp4"
        out_path = os.path.join(OUTDIR, out_name)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        # seek and write
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames_written = 0
        trail_a, trail_b = [], []

        for f in range(f0, f1+1):
            ok, frame = cap.read()
            if not ok:
                break

            def draw_track(tid, label, color, trail):
                g = tr_by_id.get(tid)
                if g is None or f not in g.index: return
                cx, cy = float(g.at[f, xcol]), float(g.at[f, ycol])
                trail.append((int(cx), int(cy)))
                if len(trail) > TRAIL: trail[:] = trail[-TRAIL:]
                for k in range(1, len(trail)):
                    cv2.line(frame, trail[k-1], trail[k], color, 2, cv2.LINE_AA)
                cv2.circle(frame, (int(cx), int(cy)), 6, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"ID {tid} {label}", (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            draw_track(id_a, la, (0,0,255), trail_a)  # red
            draw_track(id_b, lb, (255,0,0), trail_b)  # blue

            # header
            cv2.rectangle(frame, (0,0), (W, 40), (0,0,0), -1)
            txt = f"Event {i+1}/{len(top)} {r['type']}  t={t_event:.2f}s  TTC={r['ttc_s']:.2f}s  PET={r['pet_s']:.2f}s  pair:{id_a}-{id_b}"
            cv2.putText(frame, txt, (10,27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            writer.write(frame); frames_written += 1

        writer.release()

        if frames_written > 0:
            print("Saved:", out_path)
            saved += 1
        else:
            print(f"Skipped event {i} (no frames read at window {f0}-{f1}).")
            # remove empty file if created
            try:
                if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
                    os.remove(out_path)
            except Exception:
                pass
            skipped += 1

    cap.release()
    print(f"Done. Saved {saved} clips; skipped {skipped}.")
    if skipped:
        print("Tip: ensure VIDEO points to the SAME file used for tracking, and increase PRE_S/POST_S a bit.")
if __name__ == "__main__":
    main()
