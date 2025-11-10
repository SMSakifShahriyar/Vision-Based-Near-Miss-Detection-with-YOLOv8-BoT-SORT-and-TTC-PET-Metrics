# clean_traj.py
import os, pandas as pd
from collections import Counter

BASE = r"F:\MY PROJECTS\VIDEO"
RAW  = os.path.join(BASE, "traj_botsort_raw.csv")
OUT  = os.path.join(BASE, "traj_botsort_clean.csv")

DET_CONF_KEEP   = 0.50   # keep rows with conf >=
MIN_TRACK_FRAMES= 12     # drop track fragments shorter than this (≈0.4s @30fps)
MIN_AREA        = 32*32  # redundant guard

def mode_label(s):
    c = Counter([str(x) for x in s if pd.notna(x) and str(x)!=""])
    return c.most_common(1)[0][0] if c else ""

def main():
    df = pd.read_csv(RAW)
    # keep by confidence/area again (safety)
    df = df[(df["conf"] >= DET_CONF_KEEP) & (df["area"] >= MIN_AREA)].copy()
    # compute stable class PER track (majority vote)
    stable = df.groupby("id")["cls"].apply(mode_label).to_dict()
    df["cls_stable"] = df["id"].map(stable)
    # keep only long-enough tracks
    lengths = df.groupby("id")["frame"].nunique()
    good_ids = set(lengths[lengths >= MIN_TRACK_FRAMES].index)
    df = df[df["id"].isin(good_ids)].copy()
    # rename to standard columns expected by downstream code
    df = df.rename(columns={"cls_stable":"cls"})
    df = df[["id","cls","frame","t","x","y","vx","vy","speed","conf","w","h","area"]]
    df.to_csv(OUT, index=False)
    print("[DONE] cleaned trajectories ->", OUT)
    print(f"Tracks kept: {len(good_ids)}")

if __name__ == "__main__":
    main()
