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
print(find_intersections(boxes))  # -> [[0 1]]

# 3D nested layout: (n, 2, 3) = [[min], [max]]
boxes3 = np.random.rand(100_000, 2, 3)
boxes3[:, 1] = boxes3[:, 0] + 0.05
print(len(find_intersections(boxes3)))
```

Accepted layouts (input is coerced to contiguous float32):

- `(*, 4)`     — `[min_x, min_y, max_x, max_y]`
- `(*, 2, 2)`  — `[[min_x, min_y], [max_x, max_y]]`
- `(*, 6)`     — `[min_x, min_y, min_z, max_x, max_y, max_z]`
- `(*, 2, 3)`  — `[[min_x, min_y, min_z], [max_x, max_y, max_z]]`

Returns a sorted `(n, 2)` int32 array of `(i, j)` pairs with `i < j`.

## Design notes

- C++17 core (`src/cpp/`), no external runtime dependencies.
- nanobind bindings with stable ABI (`abi3`) wheels (CPython 3.12+).
- Single-threaded, CPU-only; box-box intersection queries only.
- Differences vs. the reference `lbvh` package:
  - accepts float64 input (auto-cast) in addition to float32;
  - returns deterministic, sorted pairs;
  - no `fm` dependency (AABB is inlined).

## Testing

```bash
pytest
```

Tests cross-check against a brute-force O(n^2) reference on randomized
boxes in 2D and 3D.
