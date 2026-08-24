"""LBVH: fast axis-aligned bounding box intersection.

Self-hosted reimplementation of the linear bounding volume hierarchy
algorithm (Karras 2012). Accepts 2D/3D boxes in flat or nested layout and
returns all intersecting pairs.

Examples
--------
>>> import numpy as np
>>> from lbvh2 import find_intersections
>>> boxes = np.array([[0., 0., 1., 1.], [0.5, 0.5, 1.5, 1.5]])
>>> find_intersections(boxes)
array([[0, 1]], dtype=int32)

Supported input layouts (all coerced to contiguous float32):
  - (*, 4)  : [min_x, min_y, max_x, max_y]
  - (*, 2, 2) : [[min_x, min_y], [max_x, max_y]]
  - (*, 6)  : [min_x, min_y, min_z, max_x, max_y, max_z]
  - (*, 2, 3) : [[min_x, min_y, min_z], [max_x, max_y, max_z]]
"""

from __future__ import annotations

import numpy as np

from ._lbvh2 import find_intersections as _find_intersections


def find_intersections(boxes, *, sorted: bool = False) -> np.ndarray:
    """Return an (n, 2) int32 array of intersecting box-index pairs.

    Parameters
    ----------
    boxes : array-like
        Boxes in one of the supported layouts (see module docstring).
    sorted : bool, keyword-only, default False
        If True, return pairs lexicographically sorted by (i, j).
        If False (default), return pairs in BVH traversal order — this
        matches the original `lbvh` PyPI package and is ~10-30% faster at
        large n (the sort cost dominates when there are millions of pairs).

    Returns
    -------
    np.ndarray
        Pairs (i, j) with i < j of the original indices.
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] in (4, 6):
        arr = np.ascontiguousarray(arr)
    elif arr.ndim == 3 and arr.shape[1:] in ((2, 2), (2, 3)):
        arr = np.ascontiguousarray(arr)
    else:
        raise ValueError(
            f"unsupported box layout {arr.shape}; expected (*, 4), (*, 2, 2), "
            f"(*, 6), or (*, 2, 3)"
        )
    return _find_intersections(arr, sorted)


__all__ = ["find_intersections"]
