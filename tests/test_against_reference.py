"""Side-by-side comparison against the reference `lbvh` (PyPI).

Skipped automatically if `LBVH` is not importable — install with
`pip install lbvh` to enable these tests.

Each test compares lbvh2 against the reference on the same input. Because
the reference returns pairs in BVH traversal order while lbvh2 returns
lexicographically sorted pairs, we compare as sets of (i, j) tuples.
"""

from __future__ import annotations

import numpy as np
import pytest

import lbvh2

# Skip the entire module if the reference isn't installed.
lbvh = pytest.importorskip("LBVH")


def as_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    """Convert (k,2) int array to a set of (i,j) tuples."""
    return {tuple(row) for row in pairs.tolist()}


def rand_boxes_2d(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    b = a + rng.uniform(0.0, 0.5, size=(n, 2)).astype(np.float32)
    return np.stack([a, b], axis=1)  # (n, 2, 2)


def rand_boxes_3d(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    b = a + rng.uniform(0.0, 0.5, size=(n, 3)).astype(np.float32)
    return np.stack([a, b], axis=1)  # (n, 2, 3)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 50, 500, 1000])
def test_pair_sets_match(dim: int, n: int) -> None:
    """Pair sets from lbvh2 and the reference must match on random data."""
    rng = np.random.default_rng(20240824 + n * 11 + dim)
    boxes = rand_boxes_2d(n, rng) if dim == 2 else rand_boxes_3d(n, rng)

    ours = as_set(lbvh2.find_intersections(boxes))
    theirs = as_set(lbvh.find_intersections(boxes))

    assert ours == theirs, (
        f"dim={dim} n={n}: lbvh2 has {len(ours)} pairs, reference has {len(theirs)} pairs, "
        f"|ours - theirs| = {len(ours - theirs)}, |theirs - ours| = {len(theirs - ours)}"
    )


def test_all_overlap_3d() -> None:
    """n identical boxes -> every pair intersects."""
    rng = np.random.default_rng(0)
    boxes = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32), (200, 1)
    )
    ours = as_set(lbvh2.find_intersections(boxes))
    theirs = as_set(lbvh.find_intersections(boxes))
    assert ours == theirs
    assert len(ours) == 200 * 199 // 2


def test_duplicate_morton_codes() -> None:
    """Many boxes share centroids -> identical Morton codes (tie-break path)."""
    rng = np.random.default_rng(11)
    centers = (rng.integers(0, 4, size=(300, 3)) * 0.1).astype(np.float32)
    sizes = rng.uniform(0.05, 0.15, size=(300, 3)).astype(np.float32)
    boxes = np.concatenate([centers, centers + sizes], axis=1)  # (n, 6)

    ours = as_set(lbvh2.find_intersections(boxes))
    theirs = as_set(lbvh.find_intersections(boxes))
    assert ours == theirs


def test_layout_equivalence_against_reference() -> None:
    """Reference must produce identical pair sets on flat vs nested layouts."""
    rng = np.random.default_rng(99)
    nested = rand_boxes_3d(500, rng)
    flat = np.concatenate([nested[:, 0], nested[:, 1]], axis=1)

    ref_flat = as_set(lbvh.find_intersections(flat))
    ref_nested = as_set(lbvh.find_intersections(nested))
    assert ref_flat == ref_nested  # sanity: reference itself is layout-invariant

    ours_flat = as_set(lbvh2.find_intersections(flat))
    ours_nested = as_set(lbvh2.find_intersections(nested))
    assert ours_flat == ours_nested  # lbvh2 must also be layout-invariant
    assert ours_flat == ref_flat  # and the two libraries must agree


def test_i_lt_j_invariant() -> None:
    """Reference also guarantees i < j — we double-check it here."""
    rng = np.random.default_rng(7)
    boxes = rand_boxes_3d(300, rng)
    pairs = lbvh.find_intersections(boxes)
    assert pairs.size == 0 or np.all(pairs[:, 0] < pairs[:, 1])