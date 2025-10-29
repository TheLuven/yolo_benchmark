#!/usr/bin/env python3
"""
YOLOv12 Object Tracking WITHOUT SAHI (Standard tracking for comparison).
- Uses YOLO with built-in BoT-SORT/ByteTrack tracking
- Tracks objects across video frames with persistent IDs
- For comparison with SAHI version

Usage examples:
  python yolo_tracking_standard.py --input footage/walking.mp4 --weights yolo12n.pt
  python yolo_tracking_standard.py --input footage/5.mp4 --weights yolo12n.pt
  python yolo_tracking_standard.py --input footage/walking.mp4 --weights yolo12n.pt --show --save-txt
"""
import os
import cv2
import time
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics non disponible. Installez avec: pip install ultralytics")


def find_default_video() -> Optional[str]:
    """Find a default video file in footage/ directory."""
    candidates = [
        'footage/5.mp4',
        'footage/walking.mp4',
        'footage/1.mp4',
        'footage/2.mp4',
        'footage/3.mp4',
        'footage/4.mp4',
        'footage/6.mp4',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def draw_tracks(frame: np.ndarray, boxes, track_ids, class_ids, confidences, names: List[str],
                track_history: Dict[int, List[Tuple[float, float]]], show_trails: bool = True):
    """Draw detection boxes with track IDs and optional trajectories."""
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        track_id = int(track_ids[i])
        cls = int(class_ids[i])
        conf = float(confidences[i])

        # Unique color per track ID
        np.random.seed(track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with track ID
        label_name = names[cls] if 0 <= cls < len(names) else f"cls{cls}"
        label = f"ID:{track_id} {label_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Track center point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Update track history
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))

        # Limit history length
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # Draw trajectory trail
        if show_trails and len(track_history[track_id]) > 1:
            points = np.array(track_history[track_id], dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    return frame


def track_video(input_path: str,
                weights: str = 'yolo12n.pt',
                output_path: Optional[str] = None,
                tracker: str = 'botsort.yaml',
                conf: float = 0.25,
                iou: float = 0.5,
                imgsz: int = 640,
                classes: Optional[List[int]] = None,
                show: bool = False,
                show_trails: bool = True,
                save_txt: bool = False,
                max_frames: Optional[int] = None,
                device: str = ''):
    """
    Track objects in video using YOLO with built-in tracking.

    Args:
        input_path: Path to input video
        weights: YOLO model weights path
        output_path: Path for output video
        tracker: Tracker config (botsort.yaml, bytetrack.yaml)
        conf: Confidence threshold
        iou: IOU threshold for NMS
        imgsz: Input image size
        classes: Filter by class IDs
        show: Display video during processing
        show_trails: Draw trajectory trails
        save_txt: Save tracking results to txt files
        max_frames: Limit processing to N frames
        device: Device to run on (cuda, cpu, mps)
    """
    # Ensure conf and iou are Python float (not numpy float32)
    conf = float(conf)
    iou = float(iou)
    
    # Load YOLO model
    print(f"Chargement du modèle: {weights}")
    model = YOLO(weights)

    # Get class names
    names = model.names

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Vidéo: {input_path}")
    print(f"Résolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")

    # Setup output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path('runs') / 'track_standard' / f'standard_track_{timestamp}'
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / 'tracked.mp4')
    else:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tracking state
    track_history: Dict[int, List[Tuple[float, float]]] = {}
    frame_idx = 0
    inference_times = []
    total_times = []
    detections_total = 0
    unique_tracks = set()

    # Text output
    txt_lines = []

    print("\nDémarrage du suivi vidéo STANDARD (sans SAHI)...")
    print(f"Tracker: {tracker}")
    print(f"Sortie: {output_path}")

    try:
        while True:
            t_frame0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Run tracking inference
            t0 = time.time()
            results = model.track(
                source=frame,
                persist=True,  # Keep track IDs across frames
                tracker=tracker,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                verbose=False,
                device=device,
            )
            inf_ms = (time.time() - t0) * 1000
            inference_times.append(inf_ms)

            # Extract tracking results
            annotated = frame.copy()

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                    track_ids = result.boxes.id.cpu().numpy()  # [N]
                    class_ids = result.boxes.cls.cpu().numpy()  # [N]
                    confidences = result.boxes.conf.cpu().numpy()  # [N]

                    detections_total += len(boxes)
                    unique_tracks.update(track_ids.astype(int))

                    # Draw tracking results
                    annotated = draw_tracks(
                        annotated, boxes, track_ids, class_ids, confidences,
                        names, track_history, show_trails
                    )

                    # Save tracking data to txt
                    if save_txt:
                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[i]
                            tid = int(track_ids[i])
                            cls = int(class_ids[i])
                            det_conf = float(confidences[i])  # Renamed to avoid conflict with parameter
                            # Format: frame, id, x1, y1, w, h, conf, class
                            line = f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{det_conf:.4f},{cls}\n"
                            txt_lines.append(line)

            # Add info overlay
            cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"STANDARD Detections | Tracks: {len(unique_tracks)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Inference: {inf_ms:.1f}ms",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write output
            out.write(annotated)

            # Display
            if show:
                cv2.imshow('Standard YOLO Tracking', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nArrêt demandé par l'utilisateur")
                    break

            total_ms = (time.time() - t_frame0) * 1000
            total_times.append(total_ms)
            frame_idx += 1

            # Progress
            if frame_idx % 30 == 0:
                print(f"Traité {frame_idx}/{total_frames} frames | Tracks: {len(unique_tracks)} | Avg inf: {np.mean(inference_times[-30:]):.1f}ms")

            if max_frames is not None and frame_idx >= max_frames:
                print(f"\nLimite de {max_frames} frames atteinte")
                break

    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur")
    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    # Save tracking results to txt
    if save_txt and txt_lines:
        txt_path = output_path.replace('.mp4', '_tracks.txt')
        with open(txt_path, 'w') as f:
            f.write("frame,id,x1,y1,w,h,conf,class\n")
            f.writelines(txt_lines)
        print(f"Résultats de suivi sauvegardés: {txt_path}")

    # Save statistics
    avg_inf = float(np.mean(inference_times)) if inference_times else 0.0
    avg_total = float(np.mean(total_times)) if total_times else 0.0
    elapsed_s = sum(total_times) / 1000.0 if total_times else 0.0

    stats = {
        'video': {
            'path': os.path.abspath(input_path),
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
        },
        'model': {
            'weights': weights,
            'tracker': tracker,
            'conf_threshold': conf,
            'iou_threshold': iou,
            'method': 'STANDARD (no SAHI)',
        },
        'processing': {
            'frames_processed': frame_idx,
            'detections_total': detections_total,
            'unique_tracks': len(unique_tracks),
            'elapsed_s': round(elapsed_s, 3),
            'effective_fps': round(frame_idx / max(1e-6, elapsed_s), 2),
        },
        'performance': {
            'avg_inference_ms': round(avg_inf, 2),
            'avg_total_ms': round(avg_total, 2),
            'min_inference_ms': round(float(np.min(inference_times)), 2) if inference_times else 0.0,
            'max_inference_ms': round(float(np.max(inference_times)), 2) if inference_times else 0.0,
        },
        'output': {
            'video': os.path.abspath(output_path),
        }
    }

    stats_path = output_path.replace('.mp4', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("SUIVI STANDARD TERMINÉ (sans SAHI)")
    print("="*60)
    print(f"Frames traités: {frame_idx}")
    print(f"Détections totales: {detections_total}")
    print(f"Tracks uniques: {len(unique_tracks)}")
    print(f"Temps écoulé: {elapsed_s:.2f}s")
    print(f"FPS effectif: {frame_idx / max(1e-6, elapsed_s):.2f}")
    print(f"Inference moyenne: {avg_inf:.2f}ms")
    print(f"Temps total moyen/frame: {avg_total:.2f}ms")
    print(f"\nVidéo de sortie: {output_path}")
    print(f"Statistiques: {stats_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Object Tracking STANDARD (sans SAHI) pour comparaison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default=None,
                       help='Chemin vidéo en entrée')
    parser.add_argument('--weights', type=str, default='yolo12n.pt',
                       help='Poids du modèle YOLO')
    parser.add_argument('--output', type=str, default=None,
                       help='Chemin vidéo de sortie')
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                       choices=['botsort.yaml', 'bytetrack.yaml'],
                       help='Configuration du tracker')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Seuil de confiance')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='Seuil IoU pour NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Taille de l\'image d\'entrée')
    parser.add_argument('--classes', type=int, nargs='*', default=None,
                       help='Filtrer par classes (ex: 0 pour personnes)')
    parser.add_argument('--show', action='store_true',
                       help='Afficher la vidéo pendant le traitement')
    parser.add_argument('--no-trails', action='store_true',
                       help='Désactiver l\'affichage des trajectoires')
    parser.add_argument('--save-txt', action='store_true',
                       help='Sauvegarder les résultats de suivi en TXT')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Limiter le nombre de frames (pour test)')
    parser.add_argument('--device', type=str, default='',
                       help='Device PyTorch (cuda, cpu, mps, ou vide pour auto)')

    args = parser.parse_args()

    # Find input video
    input_path = args.input or find_default_video()
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(
            "Aucune vidéo d'entrée trouvée. Spécifiez --input <chemin_video>"
        )

    track_video(
        input_path=input_path,
        weights=args.weights,
        output_path=args.output,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        show=args.show,
        show_trails=not args.no_trails,
        save_txt=args.save_txt,
        max_frames=args.max_frames,
        device=args.device,
    )


if __name__ == '__main__':
    main()
