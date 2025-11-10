import os, pandas as pd
from collections import Counter

BASE = r"F:\MY PROJECTS\VIDEO"
RAW  = os.path.join(BASE, "traj_botsort_raw.csv")
OUT  = os.path.join(BASE, "traj_botsort_clean.csv")

DET_CONF_KEEP   = 0.50
MIN_TRACK_FRAMES= 12
MIN_AREA        = 32*32

def mode_label(s):
    c = Counter([str(x) for x in s if pd.notna(x) and str(x)!=""])
    return c.most_common(1)[0][0] if c else ""

def main():
    df = pd.read_csv(RAW)
    df = df[(df["conf"] >= DET_CONF_KEEP) & (df["area"] >= MIN_AREA)].copy()
    stable = df.groupby("id")["cls"].apply(mode_label).to_dict()
    df["cls_stable"] = df["id"].map(stable)
    lengths = df.groupby("id")["frame"].nunique()
    good_ids = set(lengths[lengths >= MIN_TRACK_FRAMES].index)
    df = df[df["id"].isin(good_ids)].copy()
    df = df.rename(columns={"cls_stable":"cls"})
    df = df[["id","cls","frame","t","x","y","vx","vy","speed","conf","w","h","area"]]
    df.to_csv(OUT, index=False)
    print("[DONE] cleaned trajectories ->", OUT)
    print(f"Tracks kept: {len(good_ids)}")

if __name__ == "__main__":
    main()
