import cv2
import json
from pathlib import Path
import os

with open("outputs/matched_pairs.json", "r") as f:
    matches = json.load(f)
    
path_to_id={}
for match in matches.values():
    path_to_id[match["tacticam_path"]] = match["tacticam_id"]
    
#Steps to annotate videos
def annotate_video(video_path, meta_dir, output_path, is_tacticam):
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (
        int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Load metadata for this frame
        meta_files = list(Path(meta_dir).glob(f"frame{frame_idx}_id*.json"))
        for meta_file in meta_files:
            with open(meta_file) as f:
                data = json.load(f)

            x1, y1, x2, y2 = data["bbox"]
            box_id = data["id"]
            color = (0, 255, 0)

            # Use matched broadcast ID if tacticam, else show original
            if is_tacticam:
                crop_path = data["crop_path"]
                matched_id = path_to_id.get(crop_path, None)
                label = f"ID {matched_id}" if matched_id is not None else "Unmatched"
                color = (0, 0, 255) if matched_id is None else (255, 0, 0)
            else:
                label = f"ID {box_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved at: {output_path}")

# Annotate both videos
annotate_video("data/broadcast.mp4", "outputs/metadata", "outputs/annotated_broadcast.mp4", is_tacticam=False)
annotate_video("data/tacticam.mp4", "outputs/tacticam_metadata", "outputs/annotated_tacticam.mp4", is_tacticam=True)