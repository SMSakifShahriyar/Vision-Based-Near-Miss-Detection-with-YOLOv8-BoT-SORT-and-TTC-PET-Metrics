import os, json, cv2, numpy as np, pandas as pd

BASE = r"F:\MY PROJECTS\VIDEO"
CONF = os.path.join(BASE, "conflicts_ttc_pet.csv")
ZJS  = os.path.join(BASE, "zones.json")
OUT_CSV = os.path.join(BASE, "conflicts_with_zones.csv")
OUT_SUM = os.path.join(BASE, "conflicts_zones_summary.txt")

df = pd.read_csv(CONF)
if df.empty:
    print("No conflicts. Exiting."); raise SystemExit

zones = json.load(open(ZJS, "r"))

def in_poly(x, y, poly_pts):
    poly = np.array(poly_pts, np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def tag_zone(x, y):
    for z in zones:
        if in_poly(x, y, z["pts"]):
            return z["label"]
    return "none"


xcol = "x_px" if "x_px" in df.columns else "x"
ycol = "y_px" if "y_px" in df.columns else "y"

df["zone"] = [tag_zone(x, y) for x, y in zip(df[xcol], df[ycol])]
df.to_csv(OUT_CSV, index=False)
print("Saved per-event zones ->", OUT_CSV)

with open(OUT_SUM, "w") as f:
    f.write("Counts by zone:\n")
    for z, c in df["zone"].value_counts().items():
        f.write(f"  {z}: {c}\n")
    if "type" in df.columns:
        f.write("\nCounts by zone × type:\n")
        tab = df.pivot_table(index="zone", columns="type", values=xcol, aggfunc="count", fill_value=0)
        f.write(str(tab) + "\n")

print("Saved summary ->", OUT_SUM)
