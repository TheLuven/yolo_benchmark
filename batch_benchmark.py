from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, List

import pandas as pd


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def find_videos(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    vids: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p.resolve())
    return vids


def run_benchmark_on_video(video: Path, out_base: Path, imgsz: int | None, conf: float | None,
                            device: str | None, max_frames: int | None, models: Iterable[str] | None) -> Path:
    """
    Invoke benchmark_yolo.py for a single video, with output rooted at out_base (a directory).
    Returns the final run directory created by benchmark_yolo (timestamped), or out_base if unknown.
    """
    py = sys.executable
    bench = Path(__file__).with_name('benchmark_yolo.py')
    if not bench.exists():
        raise FileNotFoundError(f"benchmark_yolo.py not found alongside {__file__}")

    cmd: List[str] = [
        py, str(bench),
        '--video', str(video),
        '--output', str(out_base),
    ]
    if imgsz is not None:
        cmd += ['--imgsz', str(imgsz)]
    if conf is not None:
        cmd += ['--conf', str(conf)]
    if device:
        cmd += ['--device', device]
    if max_frames is not None:
        cmd += ['--max-frames', str(max_frames)]
    if models:
        cmd += ['--models', *list(models)]

    print(f"[BATCH] Running: {' '.join(cmd)}")
    # Run in blocking mode; inherit stdout/stderr for live logs
    subprocess.run(cmd, check=True)

    # The benchmark creates out_base/<timestamp>/summary.csv; detect newest
    if out_base.exists():
        candidates = sorted(out_base.glob('*/summary.csv'))
        if candidates:
            return candidates[-1].parent.resolve()
    return out_base.resolve()


def aggregate_batch(batch_root: Path, out_csv: Path) -> None:
    rows = []
    for summary in batch_root.glob('*/*/summary.csv'):
        try:
            df = pd.read_csv(summary)
            # Derive metadata: video file path from neighboring summary.json if present
            video_path = None
            json_path = summary.with_suffix('.json')
            if json_path.exists():
                # light parse without json module: read '"video": "..."'
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    video_path = j.get('video')
            df['video'] = video_path if video_path else ''
            df['run_dir'] = str(summary.parent)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {summary}: {e}")
    if rows:
        out_df = pd.concat(rows, ignore_index=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[BATCH] Aggregated summary written: {out_csv}")
    else:
        print(f"[BATCH] No summaries found to aggregate under {batch_root}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Run YOLO benchmarks on all videos in a folder.')
    ap.add_argument('--folder', type=str, default='footage', help='Folder containing input videos (default: footage)')
    ap.add_argument('--output', type=str, default=None, help='Batch output root. Default: runs/benchmarks/batch_YYYYMMDD_HHMMSS')
    ap.add_argument('--imgsz', type=int, default=None)
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--models', type=str, nargs='*', default=None, help='Optional custom models to pass through')
    args = ap.parse_args(argv)

    folder = Path(args.folder).resolve()
    vids = find_videos(folder)
    if not vids:
        print(f"[BATCH] No videos found in {folder} (extensions: {sorted(VIDEO_EXTS)})")
        return 1
    print(f"[BATCH] Found {len(vids)} videos in {folder}:")
    for v in vids:
        print(f"  - {v.name}")

    batch_root = Path(args.output).resolve() if args.output else (Path('runs') / 'benchmarks' / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    run_dirs: List[Path] = []
    for v in vids:
        video_out_base = batch_root / v.stem
        video_out_base.mkdir(parents=True, exist_ok=True)
        try:
            run_dir = run_benchmark_on_video(
                video=v,
                out_base=video_out_base,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                max_frames=args.max_frames,
                models=args.models,
            )
            run_dirs.append(run_dir)
            print(f"[BATCH] Completed: {v.name} -> {run_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[BATCH] Benchmark failed for {v.name} (exit {e.returncode})")
        except Exception as e:
            print(f"[BATCH] Error for {v.name}: {e}")

    # Aggregate all summaries from this batch
    aggregate_batch(batch_root, batch_root / 'batch_summary.csv')

    print(f"\n[BATCH] Done. Root: {batch_root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

