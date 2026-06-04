"""Analyze a capacity-scan sweep: find the knee of val LER vs model params
at each surface-code distance, fit a power-law scaling rule.

Reads run_meta.json + epochs.log from every local run directory whose
meta has the matching --tag, and extracts best val LER (mean across eval
rates) per cell. Groups by distance, prints a table, fits alpha in the
relation w*depth = alpha * d^3.

Usage:
    # Analyze a scan (default tag matches bin/qec_capacity_scan):
    python analyze_capacity_scan.py --tag capacity-scan-apr17

    # Scan a specific local directory:
    python analyze_capacity_scan.py --runs-dir ./runs/qec_runs --tag capacity-scan-apr17

    # Emit a knee-based sizing recommendation:
    python analyze_capacity_scan.py --tag capacity-scan-apr17 --recommend
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


# Parse lines like:
#   Epoch 8: loss=0.0125  p=0.001:0.01947  p=0.003:0.05601  ...
EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+):.*?loss=([0-9.eE+-]+)(.*)"
)
PRATE = re.compile(r"p=([0-9.eE+-]+):([0-9.eE+-]+)")


def read_epochs_log(path):
    """Return list of dicts: [{'epoch': int, 'loss': float, 'rates': {p: err}}, ...]"""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        m = EPOCH_LINE.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        loss = float(m.group(2))
        rates = {}
        for pm in PRATE.finditer(m.group(3)):
            rates[float(pm.group(1))] = float(pm.group(2))
        rows.append({'epoch': ep, 'loss': loss, 'rates': rates})
    return rows


def find_meta(path):
    mj = path / 'run_meta.json'
    if not mj.exists():
        return None
    with open(mj) as f:
        return json.load(f)


def best_val_ler(rows):
    """Pick checkpoint with minimum mean val LER across eval rates
    (the same selection rule eval_test_split.py uses)."""
    best = None
    best_mean = float('inf')
    for r in rows:
        if not r['rates']:
            continue
        m = sum(r['rates'].values()) / len(r['rates'])
        if m < best_mean:
            best_mean = m
            best = r
    return best, best_mean


def params_estimate(width, depth, cond_dim=32, se_ratio=8):
    """Rough parameter count for ResNetDecoder with given (w, depth).
    Good enough for plotting — exact counts come from the trained checkpoint."""
    # Two 3x3x3 convs per block + SE + FiLM cond projection
    per_block = 2 * (width * width * 27)          # conv weights
    per_block += 2 * (2 * width)                    # groupnorm
    per_block += width * max(width // se_ratio, 8)  # se fc1
    per_block += max(width // se_ratio, 8) * 2 * width  # se fc2 (dual output)
    per_block += cond_dim * 2 * width               # cond proj (FiLM scale+bias)
    per_block += 1                                  # rezero scalar
    return per_block * depth + width * 27           # + input_conv (rough)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, default="capacity-scan-apr17",
                        help="run_meta.json tag to filter on")
    parser.add_argument("--runs-dir", type=str, default="runs/qec_runs",
                        help="Root directory holding run results")
    parser.add_argument("--recommend", action="store_true",
                        help="Emit per-distance sizing recommendation")
    parser.add_argument("--out", type=str, default=None,
                        help="Write aggregate JSON to this path")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"[error] runs dir {runs_dir} not found")
        return

    # Collect all cells with the matching tag
    cells = []
    for meta_path in sorted(runs_dir.glob("*/run_meta.json")):
        meta = find_meta(meta_path.parent)
        if meta is None or meta.get('tag') != args.tag:
            continue
        rows = read_epochs_log(meta_path.parent / 'epochs.log')
        best, best_mean = best_val_ler(rows)
        cells.append({
            'dir': str(meta_path.parent),
            'distance': meta['distance'],
            'seed': meta['seed'],
            'width': meta['width'],
            'depth': meta['depth'],
            'block_type': meta['block_type'],
            'params_est': params_estimate(meta['width'], meta['depth']),
            'epochs_completed': len(rows),
            'best_epoch': best['epoch'] if best else None,
            'best_mean_val_ler': best_mean if best else None,
            'best_rates': best['rates'] if best else {},
        })

    if not cells:
        print(f"[error] no cells found with tag={args.tag} under {runs_dir}")
        return

    # Print per-distance table
    by_d = defaultdict(list)
    for c in cells:
        by_d[c['distance']].append(c)

    print(f"Capacity scan: tag={args.tag}  ({len(cells)} cells)\n")
    for d in sorted(by_d.keys()):
        group = sorted(by_d[d], key=lambda c: c['params_est'])
        print(f"=== d={d} (d^3={d**3}) ===")
        print(f"  {'w':>4} {'dep':>4} {'w*dep':>7} {'alpha=wd/d3':>12} "
              f"{'params':>10} {'epochs':>7} {'mean val LER':>14}")
        for c in group:
            wd = c['width'] * c['depth']
            alpha = wd / (d ** 3)
            ler = c['best_mean_val_ler']
            ler_s = f"{ler:.5f}" if ler is not None else "—"
            print(f"  {c['width']:>4} {c['depth']:>4} {wd:>7} "
                  f"{alpha:>12.2f} {c['params_est']:>10,} "
                  f"{c['epochs_completed']:>7} {ler_s:>14}")
        print()

    # Knee fit: for each distance, find the smallest params that achieves
    # within 1.1x of the best LER in that distance.
    if args.recommend:
        print("=== Sizing recommendation (knee: within 1.1x of best LER) ===")
        recommendations = {}
        for d in sorted(by_d.keys()):
            group = [c for c in by_d[d] if c['best_mean_val_ler'] is not None]
            if not group:
                continue
            group.sort(key=lambda c: c['params_est'])
            best_ler = min(c['best_mean_val_ler'] for c in group)
            threshold = best_ler * 1.1
            knee = next((c for c in group if c['best_mean_val_ler'] <= threshold),
                        group[-1])
            recommendations[d] = {
                'width': knee['width'],
                'depth': knee['depth'],
                'w_times_depth': knee['width'] * knee['depth'],
                'alpha': (knee['width'] * knee['depth']) / (d ** 3),
                'val_ler': knee['best_mean_val_ler'],
                'best_in_distance': best_ler,
            }
            print(f"  d={d:>2}  recommended: w={knee['width']} depth={knee['depth']}  "
                  f"(alpha={recommendations[d]['alpha']:.2f}, LER={knee['best_mean_val_ler']:.5f} "
                  f"vs best-in-d={best_ler:.5f})")

        # Fit alpha across distances
        alphas = [r['alpha'] for r in recommendations.values()]
        if alphas:
            print(f"\n  fitted alpha range: "
                  f"{min(alphas):.2f} – {max(alphas):.2f} "
                  f"(median {sorted(alphas)[len(alphas)//2]:.2f})")
            print("  suggested scaling rule: w * depth ≈ alpha * d^3")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'tag': args.tag, 'cells': cells}, f, indent=2)
        print(f"\nWrote aggregate JSON to {args.out}")


if __name__ == "__main__":
    main()
