# player-reidentification

This repository implements a complete pipeline for player tracking and re-identification across two video sources: `broadcast.mp4` and `tacticam.mp4`. The system uses YOLOv11 for detection, Deep SORT for tracking, and a cosine similarity-based embedding matcher for cross-view identity matching.

---

## Project Structure

```
Player_reid/
├── data/                  # Input videos
│   ├── broadcast.mp4
│   └── tacticam.mp4
│
├── outputs/               # Outputs from detection, tracking, and re-id
│   ├── crops/             # Cropped players from each video
│   ├── meta/              # JSON metadata with bbox + ID info per frame
│   ├── embeddings/        # Player ID-wise feature vectors
│   ├── matches.json       # ID mappings across videos
│   └── annotated/         # Final annotated videos
│
├── models/
│   └── mars-small128.pb   # Appearance encoder for Deep SORT
│
├── tracker/
│   └── deep/              # Deep SORT and detection logic
│
├── utils/
│   └── ...                # Helper scripts
│
├── main.py                # Runs tracking on broadcast.mp4
├── main_tacticam.py       # Runs tracking on tacticam.mp4
├── reid.py                # Performs cross-view re-identification
├── annotate_matched.py    # Annotates re-ID results on both videos
```

---

## Stepwise Execution

### 1. `main.py`

- Input: `data/broadcast.mp4`
- Runs YOLOv11 to detect persons, Deep SORT for tracking
- Output:
  - `outputs/crops/`: ID-wise cropped players
  - `outputs/meta/`: per-frame metadata
  - `outputs/annotated/annotated_broadcast.mp4`: tracked video

### 2. `main_tacticam.py`

- Same as above, but for `tacticam.mp4`
- Generates the same structure in `outputs/`

### 3. `reid.py`

- Inputs: embeddings from both videos
- Matches players across views using cosine similarity
- Output: `outputs/matches.json`

### 4. `annotate_matched.py`

- Input: both videos + matches
- Output: annotated versions showing matched player IDs

---

## File and Folder Summary

| File / Folder             | Purpose |
|---------------------------|---------|
| `main.py`, `main_tacticam.py` | Detect and track players |
| `outputs/crops/`          | Cropped player images per ID |
| `outputs/meta/`           | JSON metadata per frame |
| `outputs/embeddings/`     | Numpy embeddings per player |
| `outputs/matches.json`    | Mapping between broadcast and tacticam IDs |
| `outputs/annotated/`      | Final annotated videos |
| `reid.py`                 | Embedding comparison and ID linking |
| `annotate_matched.py`     | Cross-view annotated visual output |
| `models/mars-small128.pb` | Deep SORT appearance encoder |
| `data/`                   | Input videos |

---

## Approaches Tried

### Attempt 1: Re-ID Without Tracking

- Ran YOLOv11 detection and used embeddings directly on each crop
- No tracking: same player had 100+ different crops and IDs
- Result: Extremely noisy matches, re-ID failed completely

### Attempt 2: Basic Tracker (SORT or IoU-based)

- Added lightweight tracking using IoU overlap
- ID switches occurred constantly due to occlusion and motion
- Output crops were not consistent per player
- Result: ID-wise embeddings were too inconsistent

### Final Working Approach: YOLOv11 + Deep SORT

- Used YOLOv11 for person detection
- Deep SORT added temporal consistency using motion + appearance
- Resulted in consistent tracking and high-quality ID-wise embeddings
- Enabled reliable re-ID matching via cosine similarity

---

## Installation

```bash
# (Optional) Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch torchvision opencv-python ultralytics tensorflow numpy
```

---

## Notes

- `.pt` and `.mp4` files are excluded from the repo due to GitHub size limits.
- You must place the YOLOv11 model and input videos manually in the correct folders.
- The Deep SORT encoder `mars-small128.pb` must be placed in `models/`.

