import cv2
import torch
from ultralytics import YOLO
import json
from pathlib import Path

model=YOLO("models/best.pt").to("cuda" if torch.cuda.is_available() else "cpu")
#Using GTX 1650, 16GB RAM, Ryzen 5 4600H

vid_path="data/tacticam.mp4"
cap=cv2.VideoCapture(vid_path)

crop_dir = Path("outputs/tacticam_cropped_players").resolve()
meta_dir = Path("outputs/tacticam_metadata").resolve()

frame_idx = 0
id_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != 1 or conf < 0.5:  # assuming class 0 = player
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = frame[y1:y2, x1:x2]

        crop_name = f"frame{frame_idx}_id{id_counter}.jpg"
        meta = {
            "frame": frame_idx,
            "id": id_counter,
            "class": cls,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "crop_path": str(crop_dir / crop_name)
        }

        cv2.imwrite(str(crop_dir / crop_name), crop)
        with open(meta_dir / f"{crop_name}.json", "w") as f:
            json.dump(meta, f, indent=2)

        id_counter += 1

    frame_idx += 1
    if frame_idx % 5 == 0:
        print(f"Processed {frame_idx} frames")

cap.release()
