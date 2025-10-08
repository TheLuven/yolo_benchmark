import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERROR] ultralytics is required. Please install it: pip install ultralytics opencv-python", file=sys.stderr)
    raise


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_synthetic_video(dst_path: Path, num_frames: int = 240, size: Tuple[int, int] = (1280, 720), fps: float = 30.0) -> Path:
    ensure_dir(dst_path.parent)
    w, h = size
    fourcc_list = ["mp4v", "avc1", "XVID"]
    writer = None
    for four in fourcc_list:
        writer = cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*four), fps, (w, h))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        raise RuntimeError("Failed to create synthetic video writer")

    # Draw simple moving shapes
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Moving rectangles
        cv2.rectangle(frame, (50 + (i * 5) % (w - 200), 100), (150 + (i * 5) % (w - 200), 200), (0, 255, 0), -1)
        cv2.putText(frame, "SYNTH: person", (60 + (i * 5) % (w - 200), 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (300, 300 + (i * 3) % (h - 200)), (500, 450 + (i * 3) % (h - 200)), (255, 0, 0), -1)
        cv2.putText(frame, "SYNTH: car", (305, 295 + (i * 3) % (h - 200)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (800, 200), (1000, 500), (0, 0, 255), 3)
        cv2.putText(frame, f"frame {i}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        writer.write(frame)
    writer.release()
    return dst_path


def download_default_video(dst_path: Path) -> Path:
    """
    Download a default sample video with humans and vehicles if not present.
    Uses Ultralytics assets (bus.mp4). Falls back to a synthetic video if download fails.
    """
    ensure_dir(dst_path.parent)
    if dst_path.exists():
        return dst_path
    import urllib.request
    url = "https://github.com/ultralytics/assets/raw/main/videos/bus.mp4"
    try:
        print(f"[INFO] Downloading sample video to {dst_path} ...")
        urllib.request.urlretrieve(url, str(dst_path))
        return dst_path
    except Exception as e:
        print(f"[WARN] Video download failed ({e}). Generating a synthetic demo video instead.")
        return generate_synthetic_video(dst_path)


def pick_first_available_weight(candidates: List[str]) -> Optional[str]:
    """Return first available candidate path or name. If file exists locally, prefer it. Otherwise return the first name (to allow auto-download by ultralytics)."""
    # Prefer existing local files first
    for c in candidates:
        if os.path.isfile(c):
            return c
    # If none local, return the first candidate (allows ultralytics to auto-download if available)
    return candidates[0] if candidates else None


def build_default_models() -> Dict[str, str]:
    """Build default model map: friendly name -> weight path or name."""
    versions: Dict[str, List[str]] = {
        "YOLOv8": [
            "yolov8m.pt", "yolov8l.pt", "yolov8s.pt", "yolov8n.pt", "yolov8x.pt",
        ],
        "YOLOv10": [
            "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
        ],
        "YOLOv11": [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        ],
        # v12 may not exist depending on ultralytics release; we try and skip gracefully if unavailable
        "YOLOv12": [
            "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
        ],
    }
    picked: Dict[str, str] = {}
    for name, candidates in versions.items():
        w = pick_first_available_weight(candidates)
        if w is not None:
            picked[name] = w
    return picked


def open_video_reader_writer(video_path: str, out_path: Path) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    fourcc_list = ["mp4v", "avc1", "XVID"]
    writer = None
    for four in fourcc_list:
        wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*four), fps, (width, height))
        if wr.isOpened():
            writer = wr
            break
    if writer is None or (not writer.isOpened()):
        raise RuntimeError("Failed to open VideoWriter for output video")
    return cap, writer, fps, (width, height)


def time_sync():
    # High-resolution timer
    return time.perf_counter()


def run_inference_on_video(model: YOLO,
                           model_label: str,
                           video_path: str,
                           output_video_path: Path,
                           max_frames: Optional[int] = None,
                           imgsz: int = 640,
                           conf: float = 0.25,
                           device: Optional[str] = None) -> Dict:
    """
    Run inference frame-by-frame, write annotated video, and collect metrics.

    Returns a dict of metrics including per-class counts.
    """
    cap, writer, fps_src, (w, h) = open_video_reader_writer(video_path, output_video_path)

    # Prepare class name mapping after model is ready
    names = getattr(model, 'names', None)
    if isinstance(names, dict):
        class_names = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        class_names = {i: n for i, n in enumerate(names)}
    else:
        class_names = {}

    # Warmup (a couple dummy runs) for more stable timing
    warmup_runs = 2
    ret, frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise RuntimeError("Video has no frames.")

    _ = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False, device=device)
    for _i in range(max(0, warmup_runs - 1)):
        _ = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False, device=device)

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    inference_times: List[float] = []
    per_class: Dict[int, int] = {}
    total_detections = 0

    total_timer_start = time_sync()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time_sync()
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False, device=device)
        t1 = time_sync()
        dt = (t1 - t0)
        inference_times.append(dt)

        # Expect single result for one frame
        if not results:
            annotated = frame
        else:
            res = results[0]
            try:
                boxes = res.boxes
                if boxes is not None:
                    # boxes.cls: (n,) tensor of class indices
                    cls = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes, 'cls') else []
                    for c in cls:
                        per_class[c] = per_class.get(c, 0) + 1
                    total_detections += int(len(cls))
            except Exception:
                pass
            # Annotated frame
            try:
                annotated = res.plot()
            except Exception:
                annotated = frame

        # Overlay metrics
        avg_inf_ms = (np.mean(inference_times) * 1000.0) if len(inference_times) > 0 else 0.0
        text1 = f"{model_label} | frame {frame_idx} | avg inf {avg_inf_ms:.2f} ms"
        cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(annotated)

        if max_frames is not None and frame_idx >= max_frames:
            break

    total_elapsed = time_sync() - total_timer_start

    cap.release()
    writer.release()

    frames_processed = frame_idx
    if frames_processed == 0:
        raise RuntimeError("No frames processed.")

    avg_inference_ms = (np.mean(inference_times) * 1000.0) if len(inference_times) > 0 else 0.0
    throughput_fps = frames_processed / sum(inference_times) if len(inference_times) > 0 else 0.0

    per_class_named = {}
    for cid, cnt in sorted(per_class.items(), key=lambda kv: kv[0]):
        name = class_names.get(cid, str(cid))
        per_class_named[name] = int(cnt)

    metrics = {
        "model": model_label,
        "weights": getattr(model, 'ckpt_path', None) or "",
        "imgsz": imgsz,
        "conf": conf,
        "frames": frames_processed,
        "source_fps": float(fps_src),
        "total_time_sec": total_elapsed,
        "mean_inference_ms": avg_inference_ms,
        "throughput_fps": throughput_fps,
        "total_detections": int(total_detections),
        "detections_per_class": per_class_named,
        "output_video": str(output_video_path),
    }
    return metrics


def load_model(weights: str) -> YOLO:
    # ultralytics.YOLO will download if weights is a known model name with network access
    return YOLO(weights)


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO v8/v10/v11/v12 on a single video and compare metrics.")
    parser.add_argument("--video", type=str, default="", help="Path to input video. If empty, a sample video will be downloaded.")
    parser.add_argument("--output", type=str, default="runs/benchmarks", help="Output directory for results.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default=None, help="Device for inference, e.g., 'cpu', 'cuda', '0'. Default: auto by ultralytics.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process (for quick runs). Default: process full video.")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Optional custom models as weight paths or names. If omitted, defaults for v8/v10/v11/v12 are tried.")

    args = parser.parse_args()

    # Resolve output run directory
    run_dir = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(run_dir)

    # Resolve or download video
    if args.video:
        video_path = args.video
    else:
        video_path = str(download_default_video(run_dir / "input.mp4"))

    # Build models list
    model_map: Dict[str, str] = {}
    if args.models:
        # Use provided models, label them using filename or provided name order
        for idx, w in enumerate(args.models, start=1):
            label = f"Model{idx}"
            # Better label if looks like yolov.. pattern
            base = os.path.basename(w)
            label = os.path.splitext(base)[0] or label
            model_map[label] = w
    else:
        model_map = build_default_models()

    # Attempt load and benchmark each model, skipping unavailable ones
    all_metrics: List[Dict] = []
    skipped: List[Tuple[str, str]] = []

    for label, weight in model_map.items():
        print(f"\n[INFO] Loading {label}: {weight}")
        model: Optional[YOLO] = None
        try:
            model = load_model(weight)
        except Exception as e:
            print(f"[WARN] Could not load {label} ({weight}): {e}. Skipping.")
            skipped.append((label, weight))
            continue

        out_video = run_dir / f"{label}_annotated.mp4"
        try:
            metrics = run_inference_on_video(
                model=model,
                model_label=label,
                video_path=video_path,
                output_video_path=out_video,
                max_frames=args.max_frames,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
            )
            all_metrics.append(metrics)
            print(f"[DONE] {label}: throughput {metrics['throughput_fps']:.2f} FPS | mean inf {metrics['mean_inference_ms']:.2f} ms | total det {metrics['total_detections']}")
        except Exception as e:
            print(f"[WARN] Error while running {label}: {e}. Skipping.")
            skipped.append((label, weight))
            # Clean partially written file
            try:
                if out_video.exists():
                    out_video.unlink()
            except Exception:
                pass
            continue

    # Save summary
    summary_path_json = run_dir / "summary.json"
    summary_path_csv = run_dir / "summary.csv"

    with open(summary_path_json, "w", encoding="utf-8") as f:
        json.dump({"results": all_metrics, "skipped": skipped, "video": video_path}, f, indent=2, ensure_ascii=False)

    # CSV: a flat representation of key metrics
    def as_csv_row(m: Dict) -> str:
        det_classes = ";".join(f"{k}:{v}" for k, v in m.get("detections_per_class", {}).items())
        return ",".join([
            m.get("model", ""),
            str(m.get("imgsz", "")),
            str(m.get("conf", "")),
            str(m.get("frames", "")),
            f"{m.get('source_fps', 0):.2f}",
            f"{m.get('total_time_sec', 0):.4f}",
            f"{m.get('mean_inference_ms', 0):.2f}",
            f"{m.get('throughput_fps', 0):.2f}",
            str(m.get("total_detections", "")),
            f'"{det_classes}"',
            m.get("output_video", ""),
        ])

    header = ",".join([
        "model", "imgsz", "conf", "frames", "source_fps", "total_time_sec", "mean_inference_ms", "throughput_fps", "total_detections", "detections_per_class", "output_video"
    ])

    with open(summary_path_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for m in all_metrics:
            f.write(as_csv_row(m) + "\n")

    print("\n[SUMMARY]")
    for m in all_metrics:
        print(f"- {m['model']}: FPS={m['throughput_fps']:.2f}, mean_inf={m['mean_inference_ms']:.2f}ms, dets={m['total_detections']}")
        if m.get("detections_per_class"):
            # print top 5 classes by count
            top = sorted(m["detections_per_class"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            tops = ", ".join(f"{k}:{v}" for k, v in top)
            print(f"  top classes: {tops}")

    if skipped:
        print("\n[NOTE] The following models were skipped (unavailable or errored):")
        for label, w in skipped:
            print(f"  - {label}: {w}")

    print(f"\nResults saved under: {run_dir}")

    # Auto-generate plots next to the CSV using the plotting utility
    try:
        import plot_benchmarks as _pb
        print("[PLOTS] Generating comparison plots...")
        df = _pb.load_summary(summary_path_csv)
        _pb.plot_all(df, run_dir)
        print(f"[PLOTS] Saved under: {run_dir}")
    except Exception as e:
        print(f"[WARN] Could not generate plots automatically: {e}")
        print(f"       You can manually run: python plot_benchmarks.py --csv \"{summary_path_csv}\"")


if __name__ == "__main__":
    main()
