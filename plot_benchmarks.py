from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid")


def parse_detections_per_class(cell: object) -> dict[str, int]:
    """
    Parse a semicolon-separated 'k:v' string into a dict {class_name: count}.
    Robust to NaN and stray tokens; sums duplicates.
    """
    result: dict[str, int] = {}
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return result
    for item in str(cell).split(';'):
        item = item.strip()
        if not item or ':' not in item:
            continue
        k, v = item.split(':', 1)
        k = k.strip()
        try:
            v_num = int(float(v.strip()))
        except ValueError:
            continue
        result[k] = result.get(k, 0) + v_num
    return result


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Preserve CSV order for plotting categories
    if 'model' in df.columns:
        df['model'] = pd.Categorical(df['model'], categories=list(df['model']), ordered=True)
    # Backward-compat: normalize column names if needed
    rename = {}
    if 'mean_inference_time_ms' in df.columns and 'mean_inference_ms' not in df.columns:
        rename['mean_inference_time_ms'] = 'mean_inference_ms'
    if rename:
        df = df.rename(columns=rename)
    return df


def build_class_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    col = 'detections_per_class' if 'detections_per_class' in df.columns else None
    if not col:
        return pd.DataFrame(columns=['model', 'class', 'count'])
    for _, r in df.iterrows():
        per_class = parse_detections_per_class(r.get(col, ''))
        for cls, cnt in per_class.items():
            rows.append({'model': r['model'], 'class': cls, 'count': cnt})
    return pd.DataFrame(rows)


def save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_all(df: pd.DataFrame, out_dir: Path) -> None:
    # 1) Scatter: total detections vs mean_inference_ms with centered axes and descending x-axis
    req_cols = {'mean_inference_ms', 'total_detections'}
    if req_cols.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.scatterplot(
            data=df,
            x='mean_inference_ms',
            y='total_detections',
            hue='model' if 'model' in df.columns else None,
            s=90,
            ax=ax,
            legend=False,
        )
        if 'model' in df.columns:
            for _, r in df.iterrows():
                ax.annotate(str(r['model']), (r['mean_inference_ms'], r['total_detections']),
                            textcoords="offset points", xytext=(6, 5), fontsize=8)
        # Centered cross-axes: draw lines at midpoints of data ranges
        x_min, x_max = df['mean_inference_ms'].min(), df['mean_inference_ms'].max()
        y_min, y_max = df['total_detections'].min(), df['total_detections'].max()
        x_mid = x_min + (x_max - x_min) / 2.0
        y_mid = y_min + (y_max - y_min) / 2.0
        ax.axvline(x_mid, color='#9E9E9E', linestyle='--', linewidth=1)
        ax.axhline(y_mid, color='#9E9E9E', linestyle='--', linewidth=1)
        # Reverse x-axis for descending order (higher speed to the right)
        ax.invert_xaxis()
        ax.set_title('Detections vs Mean Inference Time (ms) — x descending')
        ax.set_xlabel('mean_inference_ms (lower is better)')
        ax.set_ylabel('total_detections (higher is better)')
        save_fig(fig, out_dir, 'detections_vs_inference.png')

    # 3) Bars: total detections per model
    if {'total_detections'}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(df)), 4.5))
        sns.barplot(data=df, x='model' if 'model' in df.columns else None, y='total_detections', ax=ax, color='#4C78A8')
        ax.set_title('Total Detections per Model')
        ax.set_xlabel('model')
        ax.set_ylabel('total_detections')
        if 'model' in df.columns:
            ax.tick_params(axis='x', rotation=45)
        save_fig(fig, out_dir, 'total_detections_bar.png')

    # 4) Bars: mean_inference_ms per model (lower is better)
    if {'mean_inference_ms'}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(df)), 4.5))
        sns.barplot(data=df, x='model' if 'model' in df.columns else None, y='mean_inference_ms', ax=ax, color='#F58518')
        ax.set_title('Mean Inference Time (ms) per Model (lower is better)')
        ax.set_xlabel('model')
        ax.set_ylabel('mean_inference_ms')
        if 'model' in df.columns:
            ax.tick_params(axis='x', rotation=45)
        save_fig(fig, out_dir, 'mean_inference_ms_bar.png')

    # 5) Heatmap: top-8 classes by detections across models
    cc = build_class_counts(df)
    if not cc.empty:
        # Keep model order consistent with input
        if 'model' in df.columns:
            cc['model'] = pd.Categorical(cc['model'], categories=list(df['model']), ordered=True)
        top_classes = (
            cc.groupby('class')['count']
              .sum()
              .sort_values(ascending=False)
              .head(8)
              .index
              .tolist()
        )
        cc_top = cc[cc['class'].isin(top_classes)].copy()
        heat = cc_top.pivot_table(index='model' if 'model' in cc_top.columns else None,
                                  columns='class', values='count', aggfunc='sum', fill_value=0)
        # Order columns by overall frequency (top_classes order)
        heat = heat[top_classes]

        fig_width = max(8, 1.5 * len(top_classes))
        fig_height = 0.5 * (len(heat.index) if hasattr(heat.index, '__len__') else 8) + 2
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(heat, annot=True, fmt='.0f', cmap='Blues', cbar=True, ax=ax)
        ax.set_title('Detections by Class (Top 8) per Model')
        ax.set_xlabel('class')
        ax.set_ylabel('model')
        save_fig(fig, out_dir, 'detections_by_class_heatmap_top8.png')


def find_latest_summary(default_root: Path = Path('runs') / 'benchmarks') -> Path | None:
    # Find all summary.csv one level below benchmarks
    candidates = list(default_root.glob('*/summary.csv'))
    if not candidates:
        # Also look two levels deep just in case
        candidates = list(default_root.glob('*/*/summary.csv'))
    if not candidates:
        return None
    # Prefer the lexicographically newest parent (timestamps like YYYYMMDD_HHMMSS sort correctly)
    candidates.sort(key=lambda p: str(p.parent), reverse=True)
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Plot YOLO benchmark summaries.')
    parser.add_argument('--csv', type=str, default=None, help='Path to summary.csv. If omitted, auto-detect latest under runs/benchmarks/.')
    parser.add_argument('--outdir', type=str, default=None, help='Directory to save plots. Default: alongside the CSV.')
    args = parser.parse_args(argv)

    if args.csv:
        summary_path = Path(args.csv)
    else:
        latest = find_latest_summary()
        if latest is None:
            print('No summary.csv found under runs/benchmarks/. Provide --csv path explicitly.', file=sys.stderr)
            return 2
        summary_path = latest

    if not summary_path.exists():
        print(f'Summary CSV not found: {summary_path}', file=sys.stderr)
        return 2

    out_dir = Path(args.outdir) if args.outdir else summary_path.parent

    print(f"Loading summary: {summary_path}")
    df = load_summary(summary_path)

    missing = [c for c in ['model', 'mean_inference_ms', 'total_detections'] if c not in df.columns]
    if missing:
        print(f"Warning: missing expected columns: {missing}. Some plots may be skipped.")

    plot_all(df, out_dir)
    print('All plots generated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
