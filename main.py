import cv2
import torch
from ultralytics import YOLO
from tracker.tracker import DeepSortTracker
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv11 model
model = YOLO("models/best.pt")  
model.to(device)

# Initialize Deep SORT tracker
tracker = DeepSortTracker(model_path="models/mars-small128.pb")

# Load input video
video_path = "data/broadcast.mp4"
cap = cv2.VideoCapture(video_path)

# Output video
os.makedirs("outputs/annotated", exist_ok=True)
out_path = "outputs/annotated/annotated_broadcast.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]
    bboxes = []

    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        conf = result.conf.item()
        cls = int(result.cls.item())
        
        # Filter for player class only
        if cls == 1:
            bboxes.append([x1, y1, x2, y2, conf, cls])

    # Update tracker
    tracks = tracker.update(frame, bboxes)

    # Draw tracking results
    for (bbox, track_id) in tracks:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()
print("Tracking complete. Output saved at:", out_path)
