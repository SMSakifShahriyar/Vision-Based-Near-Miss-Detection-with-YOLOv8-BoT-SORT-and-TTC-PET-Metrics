
import os
import re
import json
import cv2
import numpy as np
import pandas as pd


BASE      = r"F:\MY PROJECTS\VIDEO"
CLIP      = os.path.join(BASE, r"conflict_clips_marked\ev_06_veh-ped_t184.68_ttc1000000000.00_pet0.00.mp4")
ZONESJ    = os.path.join(BASE, "zones.json")
CONFCSV   = os.path.join(BASE, "conflicts_with_zones.csv")
OUT       = os.path.join(BASE, "cover_ev06_overlay.mp4")

TITLE     = "Surrogate Safety at Urban Intersection (Panthapath Signal)"
SUB       = "TTC/PET near-miss detection - YOLOv8 + BoT-SORT"  # ASCII only
OBS_SECONDS = 386 


PANEL_POS   = "BL"   
FONT_SCALE  = 0.9
TEXT_THICK  = 2
OUTLINE_THK = 4
PANEL_OPA   = 0.80
PANEL_PAD   = 14


ALPHA_ZONES = 0.28
LINE_THICK  = 2



def load_counts(conf_csv):
    """Load conflict summary stats from CSV if it exists."""
    if not os.path.exists(conf_csv):
        return {}, {}, 0.0
    df = pd.read_csv(conf_csv)
    if df.empty:
        return {}, {}, 0.0

    by_type = df["type"].value_counts().to_dict() if "type" in df.columns else {}
    by_zone = df["zone"].value_counts().to_dict() if "zone" in df.columns else {}
    dur_s = 0.0
    if "t" in df.columns and not df["t"].empty:
        tmin, tmax = float(df["t"].min()), float(df["t"].max())
        dur_s = max(tmax - tmin, 0.0)
    return by_type, by_zone, dur_s


def parse_event_from_name(path):
    """
    Parse event metadata from a filename like:
      ev_06_veh-ped_t184.68_ttc1000000000.00_pet0.00.mp4
    (We only use idx, etype, and t; TTC/PET are ignored.)
    """
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.fullmatch(r"ev_(\d+)_([^_]+)_t([0-9.]+)_ttc([0-9.]+)_pet([0-9.]+)", name)
    if not m:
        return {}
    idx, etype, t, _ttc, _pet = m.groups()
    try:
        return {"idx": int(idx), "etype": etype, "t": float(t)}
    except ValueError:
        return {}


def draw_translucent_polys(frame, zones, colors, alpha=0.28, edge=2):
    overlay = frame.copy()
    for i, z in enumerate(zones):
        pts = np.array(z["pts"], np.int32)
        color = colors[i % len(colors)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, (0, 0, 0), edge + 2, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], True, (255, 255, 255), edge, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def put_label_at_poly_start(frame, text, pts, color=(255, 255, 255), bg=(0, 0, 0)):
    x, y = int(pts[0][0]), int(pts[0][1])
    cv2.putText(frame, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg, 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _measure_text(s, font_scale, thickness):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return w, h


def put_panel(frame, lines, corner="TR", font_scale=0.9,
              text_thick=2, outline_thick=4, pad=14, panel_opa=0.8):
    H, W = frame.shape[:2]
    sizes = [_measure_text(s, font_scale, text_thick) for s in lines]
    maxw = max((w for w, h in sizes), default=0)
    line_h = int(max((h for w, h in sizes), default=22) * 1.4)
    box_w = maxw + 2 * pad
    box_h = line_h * len(lines) + 2 * pad

    if corner == "TL":
        x0, y0 = 12, 12
    elif corner == "TR":
        x0, y0 = W - box_w - 12, 12
    elif corner == "BL":
        x0, y0 = 12, H - box_h - 12
    else:
        x0, y0 = W - box_w - 12, H - box_h - 12

    roi = frame[y0:y0 + box_h, x0:x0 + box_w].copy()
    bg = np.full_like(roi, (0, 0, 0))
    cv2.addWeighted(bg, panel_opa, roi, 1 - panel_opa, 0, roi)
    frame[y0:y0 + box_h, x0:x0 + box_w] = roi
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1, cv2.LINE_AA)

    y = y0 + pad + line_h - int(line_h * 0.35)
    for s in lines:
        cv2.putText(frame, s, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), outline_thick, cv2.LINE_AA)
        cv2.putText(frame, s, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)
        y += line_h


def main():
    if not os.path.exists(CLIP):
        raise FileNotFoundError(f"Missing clip: {CLIP}")
    if not os.path.exists(ZONESJ):
        raise FileNotFoundError(f"Missing zones file: {ZONESJ}")

    with open(ZONESJ, "r") as f:
        zones = json.load(f)
    if not isinstance(zones, list):
        raise ValueError("zones.json must contain a list of zone dicts")
    for z in zones:
        if "pts" not in z or "label" not in z:
            raise ValueError("Each zone must have 'pts' and 'label'")

    by_type, by_zone, _dur_from_csv = load_counts(CONFCSV)
    evt = parse_event_from_name(CLIP)

    cap = cv2.VideoCapture(CLIP)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {CLIP}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    colors = [(60, 180, 255), (255, 140, 60), (60, 220, 60), (200, 160, 60), (200, 80, 200)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT, fourcc, fps, (W, H))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer for: {OUT}")

  
    lines = [TITLE, SUB, f"Observation time: {OBS_SECONDS}s"]

    if evt:
        lines += [f"Event: {evt['etype']}  t={evt['t']:.2f}s"]

    total = sum(by_type.values()) if by_type else 0
    if total > 0:
        rate_per_min = total / (OBS_SECONDS / 60.0)
        lines += [f"Total conflicts: {total}   Rate: {rate_per_min:.2f}/min"]
        parts = [f"{k}:{v}" for k, v in by_type.items()]
        lines += ["By type: " + "  ".join(parts)]
        if by_zone:
            topz = sorted(by_zone.items(), key=lambda kv: kv[1], reverse=True)[:3]
            if topz:
                lines += ["Top zones: " + "  ".join([f"{k}:{v}" for k, v in topz])]
    else:
        lines += ["(No global summary available)"]
 

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break


            draw_translucent_polys(frame, zones, colors, alpha=ALPHA_ZONES, edge=LINE_THICK)

    
            for z in zones:
                put_label_at_poly_start(frame, z["label"], z["pts"], color=(255, 255, 255), bg=(0, 0, 0))


            put_panel(
                frame,
                lines,
                corner=PANEL_POS,
                font_scale=FONT_SCALE,
                text_thick=TEXT_THICK,
                outline_thick=OUTLINE_THK,
                pad=PANEL_PAD,
                panel_opa=PANEL_OPA,
            )

            out.write(frame)
    finally:
        cap.release()
        out.release()

    print(f"[OK] Wrote cover video -> {OUT}")


if __name__ == "__main__":
    main()
