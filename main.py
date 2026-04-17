import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker (tuned)
tracker = DeepSort(
    max_age=60,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=100
)

# Load video
cap = cv2.VideoCapture("input.mp4")

# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

frame_count = 0

# For FPS
prev_time = 0

# For trajectory
track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame skipping
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # YOLO detection
    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Only detect persons
        if cls == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, 'person'))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    active_tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        active_tracks.append(track)

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)

        # Draw ID
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------- TRAJECTORY --------
        center_x = int(l + w / 2)
        center_y = int(t + h / 2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((center_x, center_y))

        # Limit trail length
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # Draw trajectory lines
        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (0, 0, 255), 2)

    # -------- PLAYER COUNT --------
    player_count = len(active_tracks)
    cv2.putText(frame, f"Players: {player_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # -------- FPS --------
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps_display)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Save frame
    out.write(frame)

    cv2.imshow("Advanced Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()