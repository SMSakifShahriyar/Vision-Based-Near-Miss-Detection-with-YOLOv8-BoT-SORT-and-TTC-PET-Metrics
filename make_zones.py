import os, json, cv2, numpy as np

BASE = r"F:\MY PROJECTS\VIDEO"
IMG  = os.path.join(BASE, "zone_frame.jpg")
OUT  = os.path.join(BASE, "zones.json")

img = cv2.imread(IMG); assert img is not None
disp = img.copy()
zones, current = [], []
label = "zone_1"
print("Instructions:")
print("  - Left-click to add points for current polygon.")
print("  - 'n' to start a NEW polygon (it will save the previous one).")
print("  - 'z' to undo last point.")
print("  - 'l' to set label for current polygon in console.")
print("  - 's' to SAVE and exit.")

def redraw():
    d = img.copy()
    # draw finished zones
    for z in zones:
        pts = np.array(z["pts"], np.int32)
        cv2.polylines(d, [pts], True, (0,255,0), 2)
        cv2.putText(d, z["label"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # draw current
    if current:
        pts = np.array(current, np.int32)
        cv2.polylines(d, [pts], False, (0,0,255), 2)
    return d

def onclick(event,x,y,flags,param):
    global current, disp
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append([int(x), int(y)])

cv2.namedWindow("zones", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("zones", onclick)

while True:
    disp = redraw()
    cv2.imshow("zones", disp)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('z') and current:
        current.pop()
    elif k == ord('l'):
        cv2.destroyWindow("zones")
        label = input("Enter label for current polygon: ").strip() or label
        cv2.namedWindow("zones", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("zones", onclick)
    elif k == ord('n'):
        if len(current) >= 3:
            zones.append({"label": label, "pts": current.copy()})
            print("Saved polygon:", label, current)
            current = []
            label = f"zone_{len(zones)+1}"
        else:
            print("Add at least 3 points before 'n'.")
    elif k == ord('s'):
        # save current if valid
        if len(current) >= 3:
            zones.append({"label": label, "pts": current.copy()})
        with open(OUT, "w") as f:
            json.dump(zones, f)
        print("Saved", OUT)
        break

cv2.destroyAllWindows()
