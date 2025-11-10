# summary_report.py — verbose summary + map + chart (+ optional PDF)
import os, json, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

BASE = r"F:\MY PROJECTS\VIDEO"
IMG  = os.path.join(BASE, "zone_frame.jpg")
ZJS  = os.path.join(BASE, "zones.json")
CONF = os.path.join(BASE, "conflicts_with_zones.csv")

OUT_MAP   = os.path.join(BASE, "zone_summary_map.png")
OUT_CHART = os.path.join(BASE, "zone_summary_chart.png")
OUT_PDF   = os.path.join(BASE, "zone_summary_report.pdf")

def main():
    print("[INFO] Loading:")
    print("  - frame:", IMG)
    print("  - zones:", ZJS)
    print("  - conflicts:", CONF)

    # --- load data ---
    assert os.path.exists(IMG), f"Missing image {IMG}"
    assert os.path.exists(ZJS), f"Missing zones {ZJS}"
    assert os.path.exists(CONF), f"Missing conflicts file {CONF}"

    zones = json.load(open(ZJS, "r"))
    df = pd.read_csv(CONF)
    if df.empty:
        print("[WARN] conflicts_with_zones.csv is empty — nothing to summarize.")
        return
    if "zone" not in df.columns:
        print("[ERROR] 'zone' column not found in conflicts CSV.")
        return

    # --- draw map overlay ---
    img = cv2.imread(IMG)
    assert img is not None, f"Cannot open {IMG}"
    overlay = img.copy()
    colors = [(0,255,255),(255,0,0),(0,255,0),(0,128,255),(255,0,255),(0,200,200)]
    for i, z in enumerate(zones):
        pts = np.array(z["pts"], np.int32)
        cv2.polylines(overlay, [pts], True, colors[i % len(colors)], 3)
        cv2.putText(overlay, z["label"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, colors[i % len(colors)], 3, cv2.LINE_AA)
    blend = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(OUT_MAP, blend)
    print("[OK] map ->", OUT_MAP)

    # --- table: counts per zone × type ---
    if "type" in df.columns:
        table = df.pivot_table(index="zone", columns="type", values="t", aggfunc="count", fill_value=0)
    else:
        table = df.groupby("zone")["t"].count().to_frame(name="total")
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum()
    print("\n[SUMMARY]\n", table, "\n")

    # --- chart ---
    ax = table.drop(index=["TOTAL"], errors="ignore").drop(columns=["total"], errors="ignore").plot(
        kind="bar", figsize=(9,5), rot=0)
    ax.set_ylabel("Conflict count")
    ax.set_title("Conflicts per zone and type")
    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=180)
    plt.close()
    print("[OK] chart ->", OUT_CHART)

    # --- optional PDF ---
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(OUT_PDF, pagesize=A4)
        # save the table as HTML and embed
        html_table = table.to_html(border=0)
        content = [
            Paragraph("<b>Intersection Conflict Analysis – Zone Summary</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph(html_table.replace("\n",""), styles["Normal"]),
            Spacer(1, 12),
            RLImage(OUT_CHART, width=500, height=300),
            Spacer(1, 12),
            RLImage(OUT_MAP, width=500, height=280)
        ]
        doc.build(content)
        print("[OK] PDF  ->", OUT_PDF)
    except Exception as e:
        print("[INFO] PDF step skipped (install 'reportlab' to enable). Reason:", e)

if __name__ == "__main__":
    main()
