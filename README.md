# Surrogate Safety Analysis at Urban Intersection (YOLOv8 + BoT-SORT)

A computer-vision pipeline for near-miss detection and surrogate safety analysis using **YOLOv8** object detection and **BoT-SORT** multi-object tracking.



##  Overview
This project estimates **surrogate safety measures** (Time-to-Collision and Post-Encroachment Time) from uncalibrated traffic video.  
Tracked trajectories are analyzed to identify and classify **near-miss events** between vehicles, pedestrians, and other road users, and summarized per intersection zone.



##  Methodology

- **Detection & Tracking**  
  Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) with the **BoT-SORT** tracker for multi-class tracking of:
  > person, bicycle, rickshaw, CNG, car, van, bus, truck

- **Trajectory Processing**  
  Cleaned and merged detections; stable class assignment per ID; dropped noisy or short-lived tracks.

- **Safety Metrics**  
  - **TTC (Time-to-Collision):** estimated under constant-velocity, approach-only model.  
  - **PET (Post-Encroachment Time):** computed at closest approach between two trajectories.  
  - Pairwise interactions gated by ≤50–60 px distance and merged within 3 s cooldown.

- **Spatial Zoning**  
  Each event is tagged to one of five manually defined polygons: **center + four approaches**, allowing per-zone summaries.



##  Repository Structure

| File | Purpose |
|------|----------|
| `track_botsort_strict.py` | Run YOLOv8 + BoT-SORT tracking and export raw trajectories |
| `clean_traj.py` | Clean trajectories, enforce stable class labels |
| `compute_ttc_pet.py` | Compute TTC & PET metrics, merge conflicts |
| `make_zones.py` | Draw & save polygon zones (interactive) |
| `conflicts_by_zones.py` | Assign conflicts to zones |
| `summary_report.py` | Generate summary charts, PDFs, and heatmaps |
| `overlay_zones_on_video.py` | Overlay zones and stats on video (for presentation) |
| `export_conflict_clips_annotated.py` | Export short clips highlighting conflicting pairs |
| `zones.json` | Saved zone definitions |
| `traj_botsort.csv` | Tracked trajectory data |
| `conflicts_ttc_pet.csv` | Detected conflict events |


## ⚙️ Pipeline Steps

```bash
# 1. Track road users
python track_botsort_strict.py

# 2. Clean & stabilize trajectories
python clean_traj.py

# 3. Compute TTC / PET conflicts
python compute_ttc_pet.py

# 4. Define and save intersection zones
python make_zones.py

# 5. Assign events to zones and summarize
python conflicts_by_zones.py
python summary_report.py

# 6. (Optional) Create annotated cover video
python overlay_zones_on_video.py
