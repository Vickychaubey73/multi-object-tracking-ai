# multi-object-tracking-ai
# Multi-Object Detection and Persistent ID Tracking

## Objective

To detect and track multiple subjects in sports/event video and assign consistent unique IDs.

## Model Used

* YOLOv8 (Ultralytics) for detection
* DeepSORT for tracking

## Why This Approach

YOLOv8 provides fast and accurate real-time detection.
DeepSORT maintains identity using motion and appearance features.

## Features

* Person detection
* Persistent ID tracking
* Trajectory visualization
* Player count
* FPS monitoring

## How to Run

pip install ultralytics opencv-python deep-sort-realtime

Place input video as `input.mp4`

Run:
python main.py

## Assumptions

* Focus on people (players) as primary subjects
* Public sports video with multiple moving individuals

## Limitations

* ID switching may occur during heavy occlusion
* Small objects like ball are not tracked

## Improvements

* Heatmap visualization
* Speed estimation
* Ball tracking

## Results

### Tracking Output
![Tracking](screenshots/img1.png)

### Trajectory Visualization
![Trajectory](screenshots/img2.png)

### Player Count & FPS
![Stats](screenshots/img3.png)
