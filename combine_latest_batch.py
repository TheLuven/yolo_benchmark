from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

# Local import of plotting utilities (reused for graphs)
import plot_benchmarks as pb


def find_latest_batch(root: Path = Path('runs') / 'benchmarks') -> Path | None:
    if not root.exists():
        return None
    batch_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith('batch_')]
    if not batch_dirs:
        return None
    batch_dirs.sort(key=lambda p: p.name, reverse=True)
    return batch_dirs[0]


def parse_counts(cell: object) -> Dict[str, int]:
    # Reuse plot_benchmarks parsing for consistency
    return pb.parse_detections_per_class(cell)


def combine_counts(series: pd.Series) -> str:
    agg: Dict[str, int] = {}
    for cell in series:
        d = parse_counts(cell)
        for k, v in d.items():
            agg[k] = agg.get(k, 0) + int(v)
    # Sort by descending count, then name
    items = sorted(agg.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ';'.join(f"{k}:{v}" for k, v in items)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0).astype(float)
    v = values.fillna(0).astype(float)
    denom = w.sum()
    if denom <= 0:
        return float(v.mean() if len(v) else 0.0)
    return float((v * w).sum() / denom)


def aggregate_batch(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns in batch_summary: model,imgsz,conf,frames,source_fps,total_time_sec,mean_inference_ms,throughput_fps,total_detections,detections_per_class,output_video,video,run_dir
    # Some may be missing in older runs; guard accordingly.
    df = df.copy()
    # Normalize column names
    for col in ['frames', 'total_detections']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    for col in ['imgsz', 'conf', 'source_fps']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['total_time_sec', 'mean_inference_ms', 'throughput_fps']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    by = 'model' if 'model' in df.columns else None
    if by is None:
        raise SystemExit('batch summary missing "model" column; cannot aggregate')

    # Aggregations per model
    groups = []
    for model, g in df.groupby(by, sort=False):
        frames_sum = int(g['frames'].sum()) if 'frames' in g.columns else 0
        total_time_sum = float(g['total_time_sec'].sum()) if 'total_time_sec' in g.columns else 0.0
        total_det_sum = int(g['total_detections'].sum()) if 'total_detections' in g.columns else 0
        mean_inf = weighted_mean(g['mean_inference_ms'], g['frames']) if 'mean_inference_ms' in g.columns else float('nan')
        # Combined throughput as total frames / total time
        throughput = (frames_sum / total_time_sum) if total_time_sum > 0 else float('nan')
        # Representative inputs: first occurrences
        imgsz = int(g['imgsz'].iloc[0]) if 'imgsz' in g.columns and pd.notna(g['imgsz'].iloc[0]) else ''
        conf = float(g['conf'].iloc[0]) if 'conf' in g.columns and pd.notna(g['conf'].iloc[0]) else ''
        # Weighted average source_fps by frames
        source_fps = weighted_mean(g['source_fps'], g['frames']) if 'source_fps' in g.columns else float('nan')
        # Counts per class aggregated
        det_pc = combine_counts(g['detections_per_class']) if 'detections_per_class' in g.columns else ''
        groups.append({
            'model': model,
            'imgsz': imgsz,
            'conf': conf,
            'frames': frames_sum,
            'source_fps': round(source_fps, 2) if pd.notna(source_fps) else '',
            'total_time_sec': round(total_time_sum, 4),
            'mean_inference_ms': round(mean_inf, 2) if pd.notna(mean_inf) else '',
            'throughput_fps': round(throughput, 2) if pd.notna(throughput) else '',
            'total_detections': total_det_sum,
            'detections_per_class': det_pc,
            'output_video': ''  # not applicable when combining across multiple videos
        })

    out_df = pd.DataFrame(groups, columns=[
        'model','imgsz','conf','frames','source_fps','total_time_sec','mean_inference_ms','throughput_fps','total_detections','detections_per_class','output_video'
    ])

    # Preserve original model order as they appeared in the batch file
    out_df['model'] = pd.Categorical(out_df['model'], categories=list(df['model'].unique()), ordered=True)
    out_df = out_df.sort_values('model').reset_index(drop=True)
    return out_df


def load_batch_table(batch_dir: Path) -> Tuple[pd.DataFrame, Optional[Path]]:
    batch_csv = batch_dir / 'batch_summary.csv'
    if batch_csv.exists():
        return pd.read_csv(batch_csv), batch_csv
    # Fallback: read all nested summary.csv files
    nested = list(batch_dir.glob('*/*/summary.csv'))
    if not nested:
        raise SystemExit(f'No batch_summary.csv or nested summary.csv files found under {batch_dir}')
    frames = [pd.read_csv(p) for p in nested]
    df = pd.concat(frames, ignore_index=True)
    return df, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Combine the latest YOLO benchmark batch and plot aggregated results.')
    ap.add_argument('--batch', type=str, default=None, help='Path to a batch_* directory. If omitted, auto-detect the latest under runs/benchmarks/.')
    ap.add_argument('--outdir', type=str, default=None, help='Directory to save the aggregated CSV and plots. Default: the batch directory itself.')
    ap.add_argument('--csv-name', type=str, default='summary.csv', help='Filename for the aggregated CSV (default: summary.csv for compatibility with plotter).')
    args = ap.parse_args(argv)

    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists() or not batch_dir.is_dir():
            print(f'Batch directory not found: {batch_dir}')
            return 2
    else:
        batch_dir = find_latest_batch()
        if batch_dir is None:
            print('No batch_* directories found under runs/benchmarks/.')
            return 2

    print(f'Using batch: {batch_dir}')
    df_raw, src_csv = load_batch_table(batch_dir)
    if src_csv:
        print(f'Loaded batch table: {src_csv}')
    else:
        print('Loaded concatenated table from nested summary.csv files')

    df_agg = aggregate_batch(df_raw)

    out_dir = Path(args.outdir) if args.outdir else batch_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.csv_name
    df_agg.to_csv(out_csv, index=False)
    print(f'Wrote aggregated summary: {out_csv}')

    # Reuse plotting util to generate the same graphs
    pb.plot_all(df_agg, out_dir)
    print('All plots generated in:', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
