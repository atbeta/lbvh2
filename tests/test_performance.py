"""Performance benchmark for lbvh2.

When the reference `LBVH` package is installed, this module also emits a
ratio of lbvh2 vs reference timings for each size. The ratios are written
to `bench-results.json` for CI tracking (a regression on the ratio is more
informative than absolute numbers).

Run directly:
    pytest tests/test_performance.py -v -k perf
or to emit JSON only:
    python tests/test_performance.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import lbvh2

# Optional reference import — ratio skipped if absent.
try:
    import LBVH as ref_lbvh  # type: ignore[import-not-found]

    HAS_REFERENCE = True
except ImportError:
    HAS_REFERENCE = False


def _rand_boxes_3d(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    b = a + rng.uniform(0.0, 0.3, size=(n, 3)).astype(np.float32)
    return np.stack([a, b], axis=1)  # (n, 2, 3)


def _time_call(fn, boxes: np.ndarray, repeats: int) -> float:
    """Average wall-clock time over `repeats` runs (seconds)."""
    # Warm-up to amortize import / first-call cost.
    fn(boxes[: min(100, len(boxes))])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(boxes)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


SIZES = [1_000, 10_000, 50_000]

# Only request the `benchmark` fixture when the plugin is actually installed.
_has_benchmark = importlib.util.find_spec("pytest_benchmark") is not None


@pytest.mark.skipif(not _has_benchmark, reason="pytest-benchmark not installed")
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sorted_out", [False, True], ids=["sorted=False", "sorted=True"])
def test_perf_lbvh2_with_pytest_benchmark(benchmark, n: int, sorted_out: bool) -> None:
    """Run with the pytest-benchmark plugin when available."""
    boxes = _rand_boxes_3d(n, seed=42)
    if HAS_REFERENCE:
        ref_lbvh.find_intersections(boxes[: min(100, n)])
    result = benchmark(lbvh2.find_intersections, boxes, sorted=sorted_out)
    assert len(result) > 0


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sorted_out", [False, True], ids=["sorted=False", "sorted=True"])
def test_perf_lbvh2_regression(n: int, sorted_out: bool) -> None:
    """Catch gross regressions: loose upper bound assertion.

    Run whether or not pytest-benchmark is installed. CI uses this for
    smoke detection; the standalone runner emits the JSON ratio.
    """
    boxes = _rand_boxes_3d(n, seed=42)
    t = _time_call(
        lambda b: lbvh2.find_intersections(b, sorted=sorted_out), boxes, repeats=3
    )
    # Loose upper bound chosen to be ~3x the observed times in CI (3.10).
    # The goal is to catch gross regressions, not to measure ms.
    upper = {1_000: 5.0, 10_000: 60.0, 50_000: 600.0}[n]
    assert t < upper, f"n={n} sorted={sorted_out}: lbvh2 took {t:.2f}s, exceeds budget {upper:.1f}s"


def _emit_json(results: list[dict]) -> None:
    out = Path(__file__).resolve().parent.parent / "bench-results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[bench] wrote {out}", file=sys.stderr)


def _run_full_bench() -> None:
    """Standalone runner: prints a table and writes JSON. Used by CI.

    Reports both the default (sorted=False) and the lexicographic (sorted=True)
    modes so the JSON can be used to track the cost of sorting.
    """
    has_ref = HAS_REFERENCE
    print(
        f"{'n':>8} {'mode':>10} {'pairs':>10} {'lbvh2_ms':>12} "
        + (f"{'ref_ms':>10} {'ratio':>8}" if has_ref else "")
    )
    print("-" * (60 if has_ref else 32))

    json_rows = []
    for n in SIZES:
        boxes = _rand_boxes_3d(n, seed=42)
        pairs = len(lbvh2.find_intersections(boxes))

        for sorted_out in (False, True):
            label = "sorted" if sorted_out else "default"
            t_us = _time_call(
                lambda b: lbvh2.find_intersections(b, sorted=sorted_out),
                boxes, repeats=3,
            ) * 1000.0

            row = {"n": n, "mode": label, "pairs": pairs, "lbvh2_ms": round(t_us, 3)}

            if has_ref:
                t_ref = _time_call(ref_lbvh.find_intersections, boxes, repeats=3) * 1000.0
                ratio = t_us / t_ref if t_ref > 0 else float("inf")
                row["ref_ms"] = round(t_ref, 3)
                row["ratio_lbvh2_over_ref"] = round(ratio, 3)
                print(
                    f"{n:>8} {label:>10} {pairs:>10} {t_us:>12.2f} "
                    f"{t_ref:>10.2f} {ratio:>8.3f}"
                )
            else:
                print(f"{n:>8} {label:>10} {pairs:>10} {t_us:>12.2f}")

            json_rows.append(row)

    print("-" * (60 if has_ref else 32))
    _emit_json(json_rows)


if __name__ == "__main__":
    _run_full_bench()