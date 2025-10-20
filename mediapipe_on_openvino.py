#!/usr/bin/env python3
"""
MediaPipe Pose Model running on OpenVINO Runtime
Extracts TFLite model from MediaPipe .task file and runs it with OpenVINO
Uses YOLO for person detection to provide ROI to MediaPipe Pose model
"""
import cv2
import numpy as np
import os
import time
import zipfile
from datetime import datetime
import argparse
from openvino.runtime import Core
from ultralytics import YOLO
import json

# MediaPipe pose landmark indices (33 keypoints)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

def extract_tflite_from_task(task_path):
    """Extract TFLite model from MediaPipe .task file"""
    print(f"Extracting TFLite model from: {task_path}")

    cache_dir = os.path.join("models", "mediapipe", "pose", "_extracted")
    os.makedirs(cache_dir, exist_ok=True)

    with zipfile.ZipFile(task_path, 'r') as z:
        # List all files
        all_files = z.namelist()
        print(f"Files in .task: {all_files}")

        # Find TFLite files
        tflite_files = [f for f in all_files if f.endswith('.tflite')]
        print(f"TFLite files found: {tflite_files}")

        if not tflite_files:
            raise RuntimeError("No .tflite file found in .task archive")

        # Use the pose landmark detector (not the detector, but the landmarks)
        tflite_file = None
        for f in tflite_files:
            if 'landmarks_detector' in f.lower() or 'landmarks' in f.lower():
                tflite_file = f
                break

        if not tflite_file:
            # Fallback: use the one with 'landmark' in name
            for f in tflite_files:
                if 'landmark' in f.lower():
                    tflite_file = f
                    break

        if not tflite_file:
            # Last resort: use largest file
            sizes = [(f, len(z.read(f))) for f in tflite_files]
            tflite_file = max(sizes, key=lambda x: x[1])[0]

        print(f"Using TFLite model: {tflite_file}")

        # Extract
        data = z.read(tflite_file)
        output_path = os.path.join(cache_dir, os.path.basename(tflite_file))

        with open(output_path, 'wb') as f:
            f.write(data)

        print(f"Extracted to: {output_path}")
        return output_path

def load_model_openvino(tflite_path):
    """Load TFLite model with OpenVINO"""
    print(f"\nLoading model with OpenVINO...")

    ie = Core()
    print(f"OpenVINO available devices: {ie.available_devices}")

    # Read TFLite model
    model = ie.read_model(model=tflite_path)

    # Get model info
    print(f"\nModel Information:")
    print(f"  Inputs: {len(model.inputs)}")
    for i, inp in enumerate(model.inputs):
        print(f"    Input {i}: shape={inp.shape}, dtype={inp.element_type}")

    print(f"  Outputs: {len(model.outputs)}")
    for i, out in enumerate(model.outputs):
        print(f"    Output {i}: shape={out.shape}, dtype={out.element_type}")

    # Compile model
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    return compiled_model

def detect_person_yolo(yolo_model, frame):
    """Detect person in frame and return bounding box"""
    results = yolo_model(frame, verbose=False, conf=0.3, classes=[0])  # class 0 = person

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    # Get the largest person detection
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)

    x1, y1, x2, y2 = boxes[largest_idx]

    # Expand box slightly for better pose detection
    w = x2 - x1
    h = y2 - y1
    margin = 0.1
    x1 = max(0, x1 - w * margin)
    y1 = max(0, y1 - h * margin)
    x2 = min(frame.shape[1], x2 + w * margin)
    y2 = min(frame.shape[0], y2 + h * margin)

    return (int(x1), int(y1), int(x2), int(y2))

def preprocess_frame(frame, input_shape, bbox=None):
    """Preprocess frame for model input"""
    # input_shape is typically [1, 256, 256, 3] for MediaPipe pose
    if len(input_shape) == 4:
        batch, height, width, channels = input_shape
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")

    # Crop to person ROI if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = frame[y1:y2, x1:x2]

    # Resize frame
    resized = cv2.resize(frame, (width, height))

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # Add batch dimension
    input_data = np.expand_dims(normalized, axis=0)

    return input_data

def decode_landmarks(outputs, frame_shape, bbox=None):
    """Decode landmarks from model outputs"""
    # MediaPipe pose model outputs multiple tensors
    # Output 0: (1, 195) - 39 landmarks with visibility
    # Output 4: (1, 117) - 39 landmarks x,y,z

    landmarks = None

    # Try output 4 first (117 values = 39 * 3)
    if len(outputs) > 4:
        output = outputs[4]
        if output.size == 117:
            flat = output.flatten()
            # 39 landmarks, take first 33
            points = flat[:99].reshape(33, 3)
            landmarks = points

    # Fallback: try to find any suitable tensor
    if landmarks is None:
        for i, output in enumerate(outputs):
            shape = output.shape
            size = np.prod(shape)

            if size >= 99:
                flat = output.flatten()

                # Try to interpret as 33 landmarks with stride 3 or 5
                for stride in [5, 4, 3]:
                    if len(flat) >= 33 * stride:
                        points = flat[:33 * stride].reshape(33, stride)

                        # Check if values look like normalized coordinates [0, 1]
                        xy = points[:, :2]
                        if np.all(xy >= -0.5) and np.all(xy <= 1.5):
                            landmarks = points
                            break

                if landmarks is not None:
                    break

    if landmarks is None:
        return None

    # Convert to pixel coordinates
    if bbox is not None:
        # Landmarks are relative to the cropped ROI
        x1, y1, x2, y2 = bbox
        roi_w = x2 - x1
        roi_h = y2 - y1
    else:
        # Landmarks are relative to full frame
        x1, y1 = 0, 0
        roi_w = frame_shape[1]
        roi_h = frame_shape[0]

    landmarks_px = []

    for i in range(33):
        x = np.clip(landmarks[i, 0], 0, 1) * roi_w + x1
        y = np.clip(landmarks[i, 1], 0, 1) * roi_h + y1

        # Get visibility if available
        visibility = landmarks[i, 3] if landmarks.shape[1] > 3 else 1.0

        landmarks_px.append({
            'x': int(x),
            'y': int(y),
            'visibility': float(visibility)
        })

    return landmarks_px

def draw_pose(frame, landmarks):
    """Draw pose skeleton on frame"""
    if landmarks is None:
        return frame

    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection

        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            # Check visibility
            if start['visibility'] > 0.5 and end['visibility'] > 0.5:
                cv2.line(frame,
                        (start['x'], start['y']),
                        (end['x'], end['y']),
                        (0, 255, 0), 2)

    # Draw keypoints
    for i, landmark in enumerate(landmarks):
        if landmark['visibility'] > 0.5:
            x, y = landmark['x'], landmark['y']

            # Different colors for different body parts
            if i < 11:  # Face/head
                color = (255, 0, 0)
            elif i < 23:  # Arms and torso
                color = (0, 255, 255)
            else:  # Legs
                color = (255, 0, 255)

            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

    return frame

def process_video(video_path, model_path, output_path=None, show=False, yolo_model_name='yolo11n.pt'):
    """Process video with MediaPipe model on OpenVINO"""

    print(f"\n{'='*70}")
    print(f"MediaPipe Pose Model on OpenVINO Runtime")
    print(f"{'='*70}")

    # Load YOLO for person detection
    print(f"\nLoading YOLO model for person detection...")
    yolo_model = YOLO(yolo_model_name)

    # Extract and load model
    tflite_path = extract_tflite_from_task(model_path)
    compiled_model = load_model_openvino(tflite_path)

    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo Information:")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Setup output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("runs", "pose_mp_ov", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "annotated.mp4")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nProcessing video...")
    print(f"Output: {output_path}\n")

    # Processing loop
    frame_count = 0
    detection_count = 0
    inference_times = []
    total_times = []
    start_time = time.time()
    last_print = start_time

    # Save landmarks data
    landmarks_data = []

    try:
        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Detect person with YOLO
            bbox = detect_person_yolo(yolo_model, frame)

            if bbox is None:
                # No person detected, write original frame
                out.write(frame)
                if show:
                    cv2.imshow('MediaPipe on OpenVINO', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_count += 1
                total_time = (time.time() - frame_start) * 1000
                total_times.append(total_time)
                inference_times.append(0)
                continue

            # Preprocess with ROI
            input_data = preprocess_frame(frame, input_shape, bbox)

            # Inference with OpenVINO
            t0 = time.time()
            result = compiled_model([input_data])
            inference_time = (time.time() - t0) * 1000
            inference_times.append(inference_time)

            # Get all outputs
            outputs = [result[output] for output in compiled_model.outputs]

            # Decode landmarks with bbox info
            landmarks = decode_landmarks(outputs, frame.shape, bbox)

            if landmarks is not None:
                detection_count += 1
                landmarks_data.append({
                    'frame': frame_count,
                    'landmarks': landmarks
                })

            # Draw pose
            annotated_frame = draw_pose(frame.copy(), landmarks)

            # Add info
            status = "Detected" if landmarks else "No detection"
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Status: {status}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Inference: {inference_time:.1f}ms",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write output
            out.write(annotated_frame)

            # Show preview
            if show:
                cv2.imshow('MediaPipe on OpenVINO', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Stats
            total_time = (time.time() - frame_start) * 1000
            total_times.append(total_time)
            frame_count += 1

            # Print progress
            now = time.time()
            if now - last_print >= 1.0:
                progress = (frame_count / total_frames) * 100
                avg_inf = np.mean(inference_times[-30:])
                avg_total = np.mean(total_times[-30:])
                eta = (total_frames - frame_count) * (avg_total / 1000.0)

                print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                      f"Inference: {avg_inf:.1f}ms | Total: {avg_total:.1f}ms | "
                      f"ETA: {eta:.1f}s", end='\r')
                last_print = now

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

        # Save statistics
        elapsed = time.time() - start_time
        avg_inference = np.mean(inference_times) if inference_times else 0
        avg_total = np.mean(total_times) if total_times else 0

        stats = {
            'video': {
                'path': os.path.abspath(video_path),
                'resolution': f"{width}x{height}",
                'fps': fps,
                'total_frames': total_frames
            },
            'processing': {
                'frames_processed': frame_count,
                'detections': detection_count,
                'detection_rate': f"{100*detection_count/max(frame_count,1):.1f}%",
                'elapsed_time': f"{elapsed:.2f}s",
                'effective_fps': f"{frame_count/max(elapsed,0.01):.2f}"
            },
            'performance': {
                'avg_inference_ms': f"{avg_inference:.2f}",
                'avg_total_ms': f"{avg_total:.2f}",
                'min_inference_ms': f"{min(inference_times):.2f}" if inference_times else 0,
                'max_inference_ms': f"{max(inference_times):.2f}" if inference_times else 0
            },
            'output': {
                'video': os.path.abspath(output_path)
            }
        }

        # Save stats
        stats_path = output_path.replace('.mp4', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save landmarks
        landmarks_path = output_path.replace('.mp4', '_landmarks.json')
        with open(landmarks_path, 'w') as f:
            json.dump(landmarks_data[:100], f, indent=2)  # Save first 100 frames

        print(f"\n\n{'='*70}")
        print(f"Processing Complete!")
        print(f"{'='*70}")
        print(f"  Frames processed: {frame_count}/{total_frames}")
        print(f"  Detections: {detection_count} ({100*detection_count/max(frame_count,1):.1f}%)")
        print(f"  Avg inference time: {avg_inference:.2f}ms")
        print(f"  Avg total time: {avg_total:.2f}ms")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Effective FPS: {frame_count/max(elapsed,0.01):.2f}")
        print(f"\n  Output video: {output_path}")
        print(f"  Statistics: {stats_path}")
        print(f"  Landmarks: {landmarks_path}")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='Run MediaPipe Pose model on OpenVINO')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str,
                       default='models/mediapipe/pose/pose_landmarker_full.task',
                       help='MediaPipe .task file')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--show', action='store_true', help='Show preview')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input not found: {args.input}")
        return

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return

    process_video(args.input, args.model, args.output, args.show)

if __name__ == '__main__':
    main()
