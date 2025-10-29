#!/usr/bin/env python3
"""
YOLOv12 Object Tracking with SAHI (Slicing Aided Hyper Inference).
- Uses SAHI for improved small object detection via image slicing
- Tracks objects across video frames with persistent IDs
- Combines sliced predictions for better accuracy

Usage examples:
  python yolo_tracking_sahi.py --input footage/walking.mp4 --weights yolo12n.pt
  python yolo_tracking_sahi.py --input footage/5.mp4 --weights yolo12n.pt --slice-size 640 --overlap 0.2
  python yolo_tracking_sahi.py --input footage/walking.mp4 --weights yolo12n.pt --show --save-txt
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

# Import SAHI first
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
except ImportError:
    raise ImportError(
        "SAHI non disponible. Installez avec: pip install sahi\n"
        "Pour une installation complète: pip install sahi[all]"
    )

# Then import Ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics non disponible. Installez avec: pip install ultralytics")
except OSError as e:
    if "DLL" in str(e):
        print("\n" + "="*60)
        print("ERREUR: Problème de DLL PyTorch détecté!")
        print("="*60)
        print("\nSolutions possibles:")
        print("1. Redémarrez votre terminal/IDE")
        print("2. Réinstallez PyTorch:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n3. Installez Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n4. Vérifiez que CUDA est bien installé (si vous utilisez GPU)")
        print("="*60 + "\n")
    raise


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


class SAHITracker:
    """Wrapper for SAHI detection + tracking."""

    def __init__(self,
                 weights: str,
                 conf_threshold: float = 0.25,
                 device: str = 'cuda:0',
                 slice_height: int = 640,
                 slice_width: int = 640,
                 overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2):
        """Initialize SAHI detection model."""
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=weights,
            confidence_threshold=conf_threshold,
            device=device,
        )

        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        # Tracking state
        self.next_track_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.max_age = 30  # frames to keep lost tracks
        self.min_hits = 3  # minimum detections before confirming track
        self.iou_threshold = 0.3  # IoU threshold for matching

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / max(union_area, 1e-6)

    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections using simple IoU matching.

        Args:
            detections: List of dicts with 'bbox' [x1,y1,x2,y2], 'score', 'category_id'

        Returns:
            List of tracked detections with added 'track_id'
        """
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]

        if len(detections) == 0:
            return []

        # Match detections to existing tracks
        matched_tracks = set()
        tracked_detections = []

        for det in detections:
            det_box = det['bbox']
            best_iou = 0
            best_track_id = None

            # Find best matching track
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue

                iou = self.compute_iou(det_box, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            # Update or create track
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = det_box
                self.tracks[best_track_id]['age'] = 0
                self.tracks[best_track_id]['hits'] += 1
                matched_tracks.add(best_track_id)

                det['track_id'] = best_track_id
                tracked_detections.append(det)
            else:
                # Create new track
                new_track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[new_track_id] = {
                    'bbox': det_box,
                    'age': 0,
                    'hits': 1,
                    'category_id': det['category_id']
                }

                det['track_id'] = new_track_id
                tracked_detections.append(det)

        # Only return confirmed tracks
        return [det for det in tracked_detections
                if self.tracks[det['track_id']]['hits'] >= self.min_hits]

    def predict_and_track(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Run SAHI prediction on frame and update tracks.

        Returns:
            (tracked_detections, inference_time_ms)
        """
        t0 = time.time()

        # Get sliced predictions from SAHI
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )

        # Convert SAHI predictions to our format
        detections = []
        for obj_pred in result.object_prediction_list:
            bbox = obj_pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
            detections.append({
                'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                'score': obj_pred.score.value,
                'category_id': obj_pred.category.id,
                'category_name': obj_pred.category.name,
            })

        inference_ms = (time.time() - t0) * 1000

        # Update tracks
        tracked_detections = self.update_tracks(detections)

        return tracked_detections, inference_ms


def draw_tracks(frame: np.ndarray,
                tracked_detections: List[Dict],
                track_history: Dict[int, List[Tuple[float, float]]],
                show_trails: bool = True):
    """Draw detection boxes with track IDs and optional trajectories."""
    for det in tracked_detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        track_id = det['track_id']
        score = det['score']
        category_name = det['category_name']

        # Unique color per track ID
        np.random.seed(track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with track ID
        label = f"ID:{track_id} {category_name} {score:.2f}"
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


def track_video_sahi(input_path: str,
                     weights: str = 'yolo12n.pt',
                     output_path: Optional[str] = None,
                     conf: float = 0.25,
                     slice_height: int = 640,
                     slice_width: int = 640,
                     overlap_ratio: float = 0.2,
                     show: bool = False,
                     show_trails: bool = True,
                     save_txt: bool = False,
                     max_frames: Optional[int] = None,
                     device: str = 'cuda:0'):
    """
    Track objects in video using SAHI + simple tracker.

    Args:
        input_path: Path to input video
        weights: YOLO model weights path
        output_path: Path for output video
        conf: Confidence threshold
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_ratio: Overlap ratio between slices
        show: Display video during processing
        show_trails: Draw trajectory trails
        save_txt: Save tracking results to txt files
        max_frames: Limit processing to N frames
        device: Device to run on (cuda:0, cpu)
    """
    # Initialize SAHI tracker
    print(f"Initialisation SAHI + Tracker avec modèle: {weights}")
    tracker = SAHITracker(
        weights=weights,
        conf_threshold=conf,
        device=device,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )

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
    print(f"SAHI - Taille des slices: {slice_width}x{slice_height}, Overlap: {overlap_ratio}")

    # Setup output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path('runs') / 'track_sahi' / f'sahi_track_{timestamp}'
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

    print("\nDémarrage du suivi vidéo avec SAHI...")
    print(f"Sortie: {output_path}")

    try:
        while True:
            t_frame0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Run SAHI prediction + tracking
            tracked_detections, inf_ms = tracker.predict_and_track(frame)
            inference_times.append(inf_ms)

            # Extract tracking results
            annotated = frame.copy()

            detections_total += len(tracked_detections)
            unique_tracks.update([det['track_id'] for det in tracked_detections])

            # Draw tracking results
            annotated = draw_tracks(
                annotated, tracked_detections, track_history, show_trails
            )

            # Save tracking data to txt
            if save_txt:
                for det in tracked_detections:
                    x1, y1, x2, y2 = det['bbox']
                    tid = det['track_id']
                    cls = det['category_id']
                    conf = det['score']
                    # Format: frame, id, x1, y1, w, h, conf, class
                    line = f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},{cls}\n"
                    txt_lines.append(line)

            # Add info overlay
            cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"SAHI Detections: {len(tracked_detections)} | Tracks: {len(unique_tracks)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Inference: {inf_ms:.1f}ms",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write output
            out.write(annotated)

            # Display
            if show:
                cv2.imshow('SAHI Tracking', annotated)
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
            'conf_threshold': conf,
            'sahi_slice_size': f"{slice_width}x{slice_height}",
            'sahi_overlap_ratio': overlap_ratio,
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
    print("SUIVI SAHI TERMINÉ")
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
        description='YOLO Object Tracking avec SAHI (Slicing Aided Hyper Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default=None,
                       help='Chemin vidéo en entrée')
    parser.add_argument('--weights', type=str, default='yolo12n.pt',
                       help='Poids du modèle YOLO')
    parser.add_argument('--output', type=str, default=None,
                       help='Chemin vidéo de sortie')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Seuil de confiance')
    parser.add_argument('--slice-height', type=int, default=640,
                       help='Hauteur des slices SAHI')
    parser.add_argument('--slice-width', type=int, default=640,
                       help='Largeur des slices SAHI')
    parser.add_argument('--overlap', type=float, default=0.2,
                       help='Ratio d\'overlap entre slices (0.0-0.5)')
    parser.add_argument('--show', action='store_true',
                       help='Afficher la vidéo pendant le traitement')
    parser.add_argument('--no-trails', action='store_true',
                       help='Désactiver l\'affichage des trajectoires')
    parser.add_argument('--save-txt', action='store_true',
                       help='Sauvegarder les résultats de suivi en TXT')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Limiter le nombre de frames (pour test)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device PyTorch (cuda:0, cpu)')

    args = parser.parse_args()

    # Find input video
    input_path = args.input or find_default_video()
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(
            "Aucune vidéo d'entrée trouvée. Spécifiez --input <chemin_video>"
        )

    track_video_sahi(
        input_path=input_path,
        weights=args.weights,
        output_path=args.output,
        conf=args.conf,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_ratio=args.overlap,
        show=args.show,
        show_trails=not args.no_trails,
        save_txt=args.save_txt,
        max_frames=args.max_frames,
        device=args.device,
    )


if __name__ == '__main__':
    main()
