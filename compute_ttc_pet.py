# compute_ttc_pet.py  — deduped, guarded TTC/PET
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import savgol_filter

# ===================== CONFIG =====================
BASE = r"F:\MY PROJECTS\VIDEO"
TRAJ_CSV = r"F:\MY PROJECTS\VIDEO\traj_botsort_clean.csv"
    # produced by your tracker
OUT_CONFLICTS = os.path.join(BASE, "conflicts_ttc_pet.csv")
OUT_SUMMARY   = os.path.join(BASE, "conflicts_summary.txt")
OUT_HEATMAP   = os.path.join(BASE, "conflict_heatmap.png")

# thresholds (tune as needed)
TTC_THRESH = 1.2  # seconds (tighter than 1.5)
PET_THRESH = 0.8     # seconds (tighter than 1.0)
LOOKAHEAD  = 6.0      # evaluate interactions up to this horizon
D_GATE_PX  = 70.0     # proximity gate (pixels) for considering a real encounter
COOLDOWN_S = 3.0      # merge triggers for same pair within this time into 1 event

# smoothing (reduces jitter)
SMOOTH_WIN = 7        # odd window size; set 0 to disable
SMOOTH_POLY= 2

# class buckets (from your model)
VEH = {"car","van","bus","truck","cng"}   # motorized
BIKE= {"bicycle","rickshaw"}              # non-motorized (adjust if needed)
PED = {"person"}

# heatmap bins
HEAT_BINS = 60
# ===================================================

def load_traj(traj_csv):
    df = pd.read_csv(traj_csv)
    # Support either schema: (x,y,vx,vy,…) or (x_px,y_px,vx_pxps,vy_pxps,…)
    if {"x","y","vx","vy"}.issubset(df.columns):
        xcol,ycol,vxcol,vycol = "x","y","vx","vy"
    elif {"x_px","y_px","vx_pxps","vy_pxps"}.issubset(df.columns):
        xcol,ycol,vxcol,vycol = "x_px","y_px","vx_pxps","vy_pxps"
    else:
        raise ValueError("Could not find expected columns. Need x/y & vx/vy in either pixel or px_* names.")
    need = ["id","cls","frame","t",xcol,ycol,vxcol,vycol]
    df = df[need].rename(columns={xcol:"x", ycol:"y", vxcol:"vx", vycol:"vy"})
    # Ensure types
    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["t"] = df["t"].astype(float)
    for c in ["x","y","vx","vy"]:
        df[c] = df[c].astype(float)
    return df

def smooth_track(track_df):
    """Savitzky–Golay smooth positions; recompute velocities."""
    if SMOOTH_WIN and len(track_df) >= SMOOTH_WIN:
        track_df = track_df.copy()
        track_df["x"] = savgol_filter(track_df["x"].values, SMOOTH_WIN, SMOOTH_POLY)
        track_df["y"] = savgol_filter(track_df["y"].values, SMOOTH_WIN, SMOOTH_POLY)
        t = track_df["t"].values
        track_df["vx"] = np.gradient(track_df["x"].values, t)
        track_df["vy"] = np.gradient(track_df["y"].values, t)
    return track_df

def interp_on(ts, t, x, y, vx, vy):
    Xi = np.interp(ts, t, x); Yi = np.interp(ts, t, y)
    Vxi = np.interp(ts, t, vx); Vyi = np.interp(ts, t, vy)
    return Xi, Yi, Vxi, Vyi

def ttc_rel_approach(p0, v0, p1, v1):
    """
    TTC for relative motion, only if approaching (dot(dp,dv) < 0).
    Returns +inf if not approaching or no positive real solution.
    """
    dp = p0 - p1
    dv = v0 - v1
    if np.dot(dp, dv) >= 0:  # not closing
        return math.inf
    a = float(np.dot(dv, dv))
    b = float(2.0 * np.dot(dp, dv))
    c = float(np.dot(dp, dp))
    if a < 1e-9:
        return math.inf
    disc = b*b - 4*a*c
    if disc < 0:
        return math.inf
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    cand = [t for t in (t1, t2) if t > 0]
    return min(cand) if cand else math.inf

def pet_from_closest_guarded(ts, Axy, Bxy, d_gate_px):
    """
    PET ≈ time difference when each agent is nearest to the other's closest-approach point.
    If min distance > d_gate_px, ignore (returns inf).
    Returns (pet, k_of_min, d_min).
    """
    d = np.linalg.norm(Axy - Bxy, axis=1)
    k = int(np.argmin(d))
    dmin = float(d[k])
    if dmin > d_gate_px:
        return math.inf, k, dmin
    pA = Axy[k]; pB = Bxy[k]
    kA = int(np.argmin(np.linalg.norm(Axy - pB, axis=1)))
    kB = int(np.argmin(np.linalg.norm(Bxy - pA, axis=1)))
    return abs(float(ts[kA] - ts[kB])), k, dmin
# put this where your helpers are
def pair_type(a_cls, b_cls, VEH, BIKE, PED):
    if a_cls in PED and b_cls in PED:
        return "ped-ped"
    if ((a_cls in VEH and b_cls in PED) or (a_cls in PED and b_cls in VEH)):
        return "veh-ped"
    if ((a_cls in VEH and b_cls in BIKE) or (a_cls in BIKE and b_cls in VEH)):
        return "veh-nmv"
    return "veh-veh"


def main():
    # Load + smooth trajectories (per ID)
    df = load_traj(TRAJ_CSV)
    by_id = {k: smooth_track(g.sort_values("t").reset_index(drop=True))
             for k,g in df.groupby("id")}
    ids = list(by_id.keys())

    # Generate raw event triggers
    events = []
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            A = by_id[ids[a_idx]]
            B = by_id[ids[b_idx]]
            # overlapping window
            t0 = max(A["t"].iloc[0], B["t"].iloc[0])
            t1 = min(A["t"].iloc[-1], B["t"].iloc[-1])
            if t1 <= t0:
                continue
            ts = np.linspace(t0, min(t0 + LOOKAHEAD, t1), 25)

            Ax, Ay, Avx, Avy = interp_on(ts, A["t"].values, A["x"].values, A["y"].values, A["vx"].values, A["vy"].values)
            Bx, By, Bvx, Bvy = interp_on(ts, B["t"].values, B["x"].values, B["y"].values, B["vx"].values, B["vy"].values)
            Axy = np.stack([Ax, Ay], axis=1)
            Bxy = np.stack([Bx, By], axis=1)

            # TTC at window start (approach-only)
            ttc = ttc_rel_approach(Axy[0], np.array([Avx[0], Avy[0]]), Bxy[0], np.array([Bvx[0], Bvy[0]]))

            # PET at closest approach with distance gate
            pet, k_closest, dmin = pet_from_closest_guarded(ts, Axy, Bxy, D_GATE_PX)

            Acls = str(A["cls"].iloc[0]); Bcls = str(B["cls"].iloc[0])
            ptype = pair_type(Acls, Bcls, VEH, BIKE, PED)
            if ptype == "ped-ped":
                continue  # skip ped–ped entirely


            serious = ((ttc < TTC_THRESH) or (pet < PET_THRESH)) and (dmin <= D_GATE_PX)
            if serious:
                events.append({
                    "pair": (ids[a_idx], ids[b_idx]),
                    "t": float(ts[0]),
                    "ttc_s": float(ttc if np.isfinite(ttc) else 1e9),
                    "pet_s": float(pet if np.isfinite(pet) else 1e9),
                    "dca_px": float(dmin),
                    "type": ptype,
                    "x_px": float((Axy[k_closest,0] + Bxy[k_closest,0]) / 2.0),
                    "y_px": float((Axy[k_closest,1] + Bxy[k_closest,1]) / 2.0),
                    "cls_a": Acls, "cls_b": Bcls,
                    "id_a": ids[a_idx], "id_b": ids[b_idx],
                })

    # Debounce/merge events per pair within COOLDOWN_S
    events.sort(key=lambda e: (e["pair"], e["t"]))
    conflicts = []
    last_time_by_pair = {}
    for e in events:
        pair = e["pair"]; t = e["t"]
        if pair in last_time_by_pair and (t - last_time_by_pair[pair]) < COOLDOWN_S:
            # merge with last conflict (same pair)
            prev = conflicts[-1]
            if prev["pair"] == pair:
                prev["ttc_s"] = min(prev["ttc_s"], e["ttc_s"])
                prev["pet_s"] = min(prev["pet_s"], e["pet_s"])
                prev["dca_px"] = min(prev["dca_px"], e["dca_px"])
                last_time_by_pair[pair] = t
                continue
        conflicts.append(e.copy())
        last_time_by_pair[pair] = t

    # Save conflicts CSV
    conf_df = pd.DataFrame(conflicts)
    if len(conf_df):
        conf_df.to_csv(OUT_CONFLICTS, index=False)
    else:
        conf_df = pd.DataFrame(columns=[
            "pair","t","ttc_s","pet_s","dca_px","type","x_px","y_px","cls_a","cls_b","id_a","id_b"
        ])
        conf_df.to_csv(OUT_CONFLICTS, index=False)

    # Summary
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"TTC_thresh={TTC_THRESH}s, PET_thresh={PET_THRESH}s, lookahead={LOOKAHEAD}s, d_gate_px={D_GATE_PX}px, cooldown={COOLDOWN_S}s\n")
        t_min = float(df["t"].min()) if len(df) else 0.0
        t_max = float(df["t"].max()) if len(df) else 0.0
        dur_min = max((t_max - t_min) / 60.0, 1e-9)
        if len(conf_df)==0:
            f.write("No serious conflicts at current settings.\n")
        else:
            f.write(f"Total serious conflicts (merged): {len(conf_df)}\n")
            by_type = conf_df["type"].value_counts().to_dict()
            f.write("By type:\n")
            for k,v in by_type.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"Duration observed: {t_max - t_min:.1f}s (~{dur_min:.2f} min)\n")
            f.write(f"Conflicts per minute: {len(conf_df)/dur_min:.2f}\n")

    # Heatmap
    if len(conf_df):
        xs, ys = conf_df["x_px"].to_numpy(), conf_df["y_px"].to_numpy()
        H, xe, ye = np.histogram2d(xs, ys, bins=HEAT_BINS)
        plt.imshow(H.T, origin="lower")
        plt.title("Conflict Density Heatmap (image coords)")
        plt.xlabel("x (px)"); plt.ylabel("y (px)")
        plt.tight_layout(); plt.savefig(OUT_HEATMAP, dpi=180); plt.close()

    print(f"[OK] Conflicts CSV  -> {OUT_CONFLICTS}")
    print(f"[OK] Summary        -> {OUT_SUMMARY}")
    if len(conf_df):
        print(f"[OK] Heatmap PNG    -> {OUT_HEATMAP}")
    else:
        print("[OK] No conflicts found with current thresholds.")

if __name__ == "__main__":
    main()
