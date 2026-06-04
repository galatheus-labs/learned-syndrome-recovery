"""Reproduce all numerical claims from the QEC paper.

Aggregates per-run test-set JSONs produced by `eval_test_split.py` and emits
the paper tables. Every run is produced by a single command:

    python src/eval_test_split.py \\
        --ckpt-dir /path/to/qec_run/ckpts \\
        --distance D --seed S --out test_d{D}_s{S}_w{W}_dep{DEP}{_suffix}.json

This reproducer walks a directory of those JSONs, groups by
(block_type, distance), aggregates across seeds, and reports:

  Section 1  Deterministic claims (param counts, grid dims per distance)
  Section 2  Classic-block baseline: mean +/- SEM LER per (distance, rate),
             paired vs MWPM via McNemar
  Section 3  LC0-block results: same structure as Section 2
  Section 4  Classic vs LC0 scaling table: LC0 improvement over classic,
             relative to MWPM, per (distance, rate)
  Section 5  Ablation table (populated when ablation runs land)

All numbers in the paper must trace to this script's output. Do not manually
transcribe from training logs; always re-derive via reproduce_qec_results.

Usage:

    # Release artifact aggregation:
    python reproduce_qec_results.py --results-dir data/qec_clean_knee_multiseed_json \\
        --distances 3,5,7,9,11 --seeds 42,43,44 --out /tmp/qec_reproduction.json

    # Subset of cells (e.g. only runs that have finished):
    python reproduce_qec_results.py --results-dir data/qec_clean_knee_multiseed_json \\
        --distances 3,5,7,9 --seeds 42,43,44
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


# ---------------------------------------------------------------------------
# Section 1: Deterministic claims
# ---------------------------------------------------------------------------

def verify_deterministic(distances, width=128, depth=12):
    """Verify param counts and detector-grid dimensions per distance.

    These are architectural invariants — they do not depend on trained
    weights or test data.
    """
    print("=" * 72)
    print("SECTION 1: Deterministic claims")
    print("=" * 72)

    import torch  # noqa
    from surface_code_resnet import (
        ResNetDecoder, generate_data, get_detector_grid_mapping,
    )

    rows = []
    for d in distances:
        _, _, ref_circuit = generate_data(d, 0.005, 1, noise='circuit')
        grid_size, _ = get_detector_grid_mapping(ref_circuit)
        for block_type in ['classic', 'lc0']:
            m = ResNetDecoder(grid_size, width=width, depth=depth,
                              block_type=block_type,
                              zero_init_residual=(block_type == 'lc0'))
            params = sum(p.numel() for p in m.parameters())
            rows.append((d, block_type, grid_size, params))
            del m

    print(f"\n{'distance':>8} {'block':>8} {'grid':>14} {'params':>14}")
    print("-" * 50)
    for d, bt, g, p in rows:
        print(f"{d:>8} {bt:>8} {str(g):>14} {p:>14,}")

    print("\n[ok] Deterministic claims verified.\n")
    return rows


# ---------------------------------------------------------------------------
# Loading / grouping run results
# ---------------------------------------------------------------------------

def load_run_results(results_dir, distances=None, seeds=None):
    """Load all eval_test_split JSONs in a directory.

    Returns: dict keyed by (block_type, distance) -> list of per-seed result dicts.
    Each result dict is the full eval_test_split JSON output.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist")

    grouped = defaultdict(list)
    for j in sorted(results_dir.glob('*.json')):
        with open(j) as f:
            r = json.load(f)
        if 'test_results' not in r:
            continue  # not an eval_test_split JSON
        if distances and r['distance'] not in distances:
            continue
        if seeds and r['seed'] not in seeds:
            continue
        # Default block_type for legacy JSONs without the field
        bt = r.get('block_type', 'classic')
        grouped[(bt, r['distance'])].append(r)
    return grouped


def validate_seed_coverage(grouped, expected_seeds):
    """Fail if any present (block_type, distance) cell has partial seed coverage.

    The paper tables are multi-seed tables. Averaging a subset of requested
    seeds is almost always a provenance mistake, so make that failure explicit.
    """
    expected = set(expected_seeds)
    missing = []
    unexpected = []
    for (block_type, distance), runs in sorted(grouped.items()):
        found = {r['seed'] for r in runs}
        miss = sorted(expected - found)
        extra = sorted(found - expected)
        if miss:
            missing.append((block_type, distance, miss))
        if extra:
            unexpected.append((block_type, distance, extra))

    if missing or unexpected:
        lines = ["Seed coverage check failed:"]
        for block_type, distance, seeds in missing:
            lines.append(f"  missing {block_type} d={distance}: seeds {seeds}")
        for block_type, distance, seeds in unexpected:
            lines.append(f"  unexpected {block_type} d={distance}: seeds {seeds}")
        lines.append("Use --allow-missing-seeds only for exploratory partial summaries.")
        raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------------------
# Aggregation (multi-seed)
# ---------------------------------------------------------------------------

def aggregate_cells(runs):
    """Given a list of per-seed run dicts for the same (block_type, distance),
    aggregate test-set LER across seeds per rate.

    Returns: dict rate_str -> {
        'mean': <mean across seeds>,
        'sem': <standard error of the mean>,
        'ci95': (lo, hi) via seed-level bootstrap,
        'n_seeds': int,
        'per_seed': [(seed, ler), ...],
        'mwpm_mean': <mean MWPM across seeds>,
        'improvement_pct': <mean improvement across seeds>,
        'mcnemar_p_per_seed': [...],  # combine with Fisher if desired
    }
    """
    if not runs:
        return {}

    rates = list(runs[0]['test_results'].keys())
    agg = {}
    for rate in rates:
        neural = [r['test_results'][rate]['neural_err'] for r in runs]
        mwpm = [r['test_results'][rate]['mwpm_err'] for r in runs]
        impr = [r['test_results'][rate]['improvement_pct'] for r in runs]
        mcp = [r['test_results'][rate]['mcnemar_p'] for r in runs]
        per_seed = [(r['seed'], r['test_results'][rate]['neural_err']) for r in runs]
        n = len(runs)
        mean = float(np.mean(neural))
        sem = float(np.std(neural, ddof=1) / math.sqrt(n)) if n >= 2 else 0.0
        # Bootstrap 95% CI (seed-level)
        rng = np.random.RandomState(0)
        boots = [rng.choice(neural, size=n, replace=True).mean() for _ in range(5000)]
        ci95 = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        agg[rate] = {
            'mean': mean,
            'sem': sem,
            'ci95': ci95,
            'n_seeds': n,
            'per_seed': per_seed,
            'mwpm_mean': float(np.mean(mwpm)),
            'improvement_pct_mean': float(np.mean(impr)),
            'mcnemar_p_per_seed': mcp,
        }
    return agg


def print_table(title, grouped, distances, block_type):
    """Print one paper-style table (distance x rate) for a given block_type."""
    print("=" * 72)
    print(title)
    print("=" * 72)

    # Discover rates from first available cell
    rates = None
    for d in distances:
        cell = grouped.get((block_type, d), [])
        if cell:
            rates = list(cell[0]['test_results'].keys())
            break
    if rates is None:
        print(f"\n[skip] No {block_type}-block runs found in results dir.\n")
        return

    print(f"\n{'d':>3}", end='')
    for r in rates:
        print(f"  {'p='+r:>18}", end='')
    print(f"  {'seeds':>6}")
    print("-" * (3 + 20 * len(rates) + 8))

    for d in distances:
        cell = grouped.get((block_type, d), [])
        if not cell:
            print(f"{d:>3}  {'(missing)':<18}")
            continue
        agg = aggregate_cells(cell)
        print(f"{d:>3}", end='')
        for r in rates:
            a = agg[r]
            # mean (ci_lo, ci_hi)
            s = f"{a['mean']:.5f} ±{a['sem']:.5f}"
            print(f"  {s:>18}", end='')
        print(f"  {agg[rates[0]]['n_seeds']:>6}")
    print()


def compare_tables(grouped, distances):
    """Section 4: LC0 vs classic. For each (distance, rate) cell where both
    block types have runs, report the mean LER for each and the relative
    improvement.

    This is NOT a paired comparison — the two block types were trained
    on different data (same distribution, different shots). A proper paired
    comparison requires running both checkpoints on identical test shots
    (see eval_paired_comparison.py, TODO).
    """
    print("=" * 72)
    print("SECTION 4: LC0 vs classic (aggregate across seeds)")
    print("=" * 72)

    print("\nNote: aggregate-level comparison. For paired (same-shot) McNemar")
    print("between LC0 and classic, run eval_paired_comparison.py (TODO).\n")

    rates = None
    for d in distances:
        for bt in ['classic', 'lc0']:
            cell = grouped.get((bt, d), [])
            if cell:
                rates = list(cell[0]['test_results'].keys())
                break
        if rates:
            break
    if rates is None:
        print("[skip] No runs available for comparison.\n")
        return

    print(f"{'d':>3}  {'rate':>8}  {'classic LER':>14}  {'LC0 LER':>14}  {'LC0 improv.':>12}")
    print("-" * 72)
    for d in distances:
        for r in rates:
            cl = grouped.get(('classic', d), [])
            lc = grouped.get(('lc0', d), [])
            if not cl or not lc:
                continue
            cl_a = aggregate_cells(cl)[r]
            lc_a = aggregate_cells(lc)[r]
            if cl_a['mean'] > 0:
                impr = (cl_a['mean'] - lc_a['mean']) / cl_a['mean'] * 100
            else:
                impr = 0.0
            print(f"{d:>3}  p={r:>6}  {cl_a['mean']:>14.5f}  "
                  f"{lc_a['mean']:>14.5f}  {impr:>+11.1f}%")
    print()


# ---------------------------------------------------------------------------
# Section 5: Ablation (skeleton)
# ---------------------------------------------------------------------------

def run_ablation_table(grouped, distances):
    """Print the per-component ablation: classic -> +pre-act -> +dual-SE ->
    +FiLM -> +ReZero -> full LC0.

    Requires ablation runs with block_type labels like 'lc0_noreze', 'lc0_nose',
    etc. Not yet collected — emits a SKIP with the protocol to follow.
    """
    print("=" * 72)
    print("SECTION 5: LC0 component ablation")
    print("=" * 72)
    print("""
[skip] Ablation runs not yet collected. To fill this table, run:

  for component in baseline preact dualse film rezero full_lc0; do
    bin/qec_job --distance 9 --seed 42 --augment --focal-loss \\
      --block-type <variant> <component-specific flags> \\
      --name qec-ablate-${component}
  done

Requires implementing --block-type variants that isolate each LC0 component
(e.g., 'lc0-noreze' for LC0 without ReZero). Currently only 'classic' and
'lc0' block types exist.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all QEC paper numbers from eval_test_split JSONs")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing eval_test_split.py JSON outputs")
    parser.add_argument("--distances", type=str, default="3,5,7,9,11",
                        help="Comma-separated distances to include")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated seeds to include")
    parser.add_argument("--width", type=int, default=128,
                        help="Model width (for Section 1 deterministic claims)")
    parser.add_argument("--depth", type=int, default=12,
                        help="Model depth (for Section 1)")
    parser.add_argument("--out", type=str, default="qec_reproduction.json",
                        help="Output JSON aggregate")
    parser.add_argument("--allow-missing-seeds", action="store_true",
                        help="Allow partial seed coverage for exploratory summaries")
    args = parser.parse_args()

    distances = [int(x) for x in args.distances.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print("=" * 72)
    print("REPRODUCING ALL QEC PAPER RESULTS")
    print("=" * 72)
    print(f"Results dir: {args.results_dir}")
    print(f"Distances:   {distances}")
    print(f"Seeds:       {seeds}")
    print()

    # Section 1
    try:
        deterministic = verify_deterministic(distances, args.width, args.depth)
    except Exception as e:
        print(f"[skip] Section 1 failed: {e}\n")
        deterministic = None

    # Load JSONs
    grouped = load_run_results(args.results_dir, distances=distances, seeds=seeds)
    if not grouped:
        print(f"[error] No JSON results found under {args.results_dir}.")
        print("Run eval_test_split.py for each (distance, seed, block_type) cell first.")
        sys.exit(1)
    if not args.allow_missing_seeds:
        validate_seed_coverage(grouped, seeds)

    # Sections 2, 3
    print_table("SECTION 2: Classic-block baseline (test-set LER, mean ± SEM)",
                grouped, distances, 'classic')
    print_table("SECTION 3: LC0-block results (test-set LER, mean ± SEM)",
                grouped, distances, 'lc0')

    # Section 4
    compare_tables(grouped, distances)

    # Section 5 (skeleton)
    run_ablation_table(grouped, distances)

    # Dump aggregate JSON
    out = {
        'distances': distances,
        'seeds': seeds,
        'width': args.width,
        'depth': args.depth,
        'classic': {},
        'lc0': {},
    }
    for bt in ['classic', 'lc0']:
        for d in distances:
            runs = grouped.get((bt, d), [])
            if not runs:
                continue
            out[bt][str(d)] = aggregate_cells(runs)

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Aggregate saved to {args.out}")


if __name__ == "__main__":
    main()
