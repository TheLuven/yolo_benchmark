#!/usr/bin/env python3
"""
Debug script to check landmark coordinates
"""
import json
import cv2

# Load landmarks
with open('runs/pose_mp_ov/20251020_172442/annotated_landmarks.json', 'r') as f:
    data = json.load(f)

# Check first frame with detection
if len(data) > 0:
    frame_data = data[0]
    print(f"Frame {frame_data['frame']}:")
    print(f"Number of landmarks: {len(frame_data['landmarks'])}")
    
    # Show first few landmarks
    for i in range(min(5, len(frame_data['landmarks']))):
        lm = frame_data['landmarks'][i]
        print(f"  Landmark {i}: x={lm['x']}, y={lm['y']}, z={lm.get('z', 'N/A')}, vis={lm['visibility']:.3f}")
    
    # Check landmark distribution
    xs = [lm['x'] for lm in frame_data['landmarks']]
    ys = [lm['y'] for lm in frame_data['landmarks']]
    
    print(f"\nX coordinates: min={min(xs)}, max={max(xs)}, range={max(xs)-min(xs)}")
    print(f"Y coordinates: min={min(ys)}, max={max(ys)}, range={max(ys)-min(ys)}")
    
    print("\nExpected video dimensions: 768x432")
    print("If landmarks are all in top-left corner (e.g., all < 100), there's a projection bug")

# Load video to check a frame visually
cap = cv2.VideoCapture('footage/walking.mp4')
ret, frame = cap.read()
if ret:
    print(f"\nActual frame shape: {frame.shape} (height x width x channels)")
cap.release()

