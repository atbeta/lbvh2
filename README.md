# lbvh2

Self-hosted reimplementation of the **LBVH** (linear bounding volume
hierarchy) algorithm for fast axis-aligned bounding box (AABB) intersection
pair finding.

Algorithm reference (public):

- T. Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
  and k-d Trees", High Performance Graphics 2012.
- NVIDIA Developer Blog, "Thinking Parallel" (Parts I–III).

## Migrating from the reference `lbvh`

See [docs/migration-from-lbvh.md](docs/migration-from-lbvh.md) for a
side-by-side comparison and a drop-in replacement guide.

## Install (from source)

```bash
pip install .
```

## Usage

```python
import numpy as np
from lbvh2 import find_intersections

# 2D flat layout: (n, 4) = [min_x, min_y, max_x, max_y]
boxes = np.array([
    [0.0, 0.0, 1.0, 1.0],
    [0.5, 0.5, 1.5, 1.5],
    [1.4, 1.4, 2.4, 2.4],
])
print(find_intersections(boxes))  # -> [[0 1],[1 2]]

# 3D nested layout: (n, 2, 3) = [[min], [max]]
boxes3 = np.random.rand(100_000, 2, 3)
boxes3[:, 1] = boxes3[:, 0] + 0.05
print(len(find_intersections(boxes3)))

# Sorted output (deterministic, ~10-30% slower at large n):
print(find_intersections(boxes, sorted=True))
```

Accepted layouts (input is coerced to contiguous float32):

- `(*, 4)`     — `[min_x, min_y, max_x, max_y]`
- `(*, 2, 2)`  — `[[min_x, min_y], [max_x, max_y]]`
- `(*, 6)`     — `[min_x, min_y, min_z, max_x, max_y, max_z]`
- `(*, 2, 3)`  — `[[min_x, min_y, min_z], [max_x, max_y, max_z]]`

Returns an `(n, 2)` int32 array of `(i, j)` pairs with `i < j`. By default
pairs are emitted in BVH traversal order (matches the reference `lbvh`
package); pass `sorted=True` for lexicographically sorted output.

Requires **Python ≥ 3.10** (lowered from 3.12 once we verified the source
needs no 3.12-only features; only the abi3 wheel tag changed).

## Design notes

- C++17 core (`src/cpp/`), no external runtime dependencies.
- nanobind 2.x bindings; per-Python-version wheels (`cp310` / `cp311` /
  `cp312`). We do **not** ship abi3 wheels — nanobind 2.x requires
  cp312+ for abi3, and we want to support 3.10 too.
- Cross-platform: builds and tests pass on Linux (GCC/Clang), macOS
  (Apple Clang), and Windows (MSVC 19.14+). The C++ core uses portable
  bitwise leading-zero count instead of `__builtin_clz` so MSVC compiles
  cleanly. CI matrix covers `ubuntu-latest` and `windows-latest`.
- Single-threaded, CPU-only; box-box intersection queries only.
- Differences vs. the reference `lbvh` package:
  - accepts float64 input (auto-cast) in addition to float32;
  - returns BVH-traversal-order pairs by default; pass `sorted=True` for
    lexicographically sorted output (the reference is always unsorted);
  - no `fm` dependency (AABB is inlined).

## Output order: `sorted` parameter

```python
find_intersections(boxes)               # default: BVH traversal order
find_intersections(boxes, sorted=True)  # lex order, deterministic
```

The default matches the reference `lbvh` for drop-in replacement; users
who want determinism or easier unit testing can opt into `sorted=True`
at a measured ~10-30% slowdown for n ≥ 10k (see
`tests/test_performance.py` for the live ratio).

## Testing

```bash
pytest
```

The test suite has two parts:

1. **Brute-force cross-check** (`tests/test_lbvh2.py`) — randomized 2D/3D
   boxes compared against an O(n^2) reference.
2. **Reference comparison** (`tests/test_against_reference.py`) — runs the
   same inputs through the original PyPI `lbvh` package and compares pair
   sets. Skipped automatically unless `pip install lbvh` is run first
   (the reference only ships wheels up to cp311).
3. **Performance benchmark** (`tests/test_performance.py`) — emits a JSON
   ratio of lbvh2 vs reference timings at n = 1k/10k/50k.

Run all three with:

```bash
pip install lbvh  # enables tests/test_against_reference.py
pytest tests/
python tests/test_performance.py  # writes bench-results.json
```
