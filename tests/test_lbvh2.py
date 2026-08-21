import numpy as np
import pytest

from lbvh2 import find_intersections


def brute_force(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    pairs = []
    n = len(boxes)

    def lo(b):
        return boxes[b, 0]
    def hi(b):
        return boxes[b, 1]

    for i in range(n):
        for j in range(i + 1, n):
            if np.all(lo(i) <= hi(j)) and np.all(lo(j) <= hi(i)):
                pairs.append([i, j])
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def rand_boxes_2d(n, rng):
    a = rng.uniform(-1.0, 1.0, size=(n, 2))
    b = a + rng.uniform(0.0, 0.5, size=(n, 2))
    return np.stack([a, b], axis=1)  # (n, 2, 2)


def rand_boxes_3d(n, rng):
    a = rng.uniform(-1.0, 1.0, size=(n, 3))
    b = a + rng.uniform(0.0, 0.5, size=(n, 3))
    return np.stack([a, b], axis=1)  # (n, 2, 3)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 50, 500, 1000])
def test_correctness(dim, n):
    rng = np.random.default_rng(12345)
    boxes = rand_boxes_2d(n, rng) if dim == 2 else rand_boxes_3d(n, rng)
    got = find_intersections(boxes)
    expected = brute_force(boxes)
    np.testing.assert_array_equal(got, expected)


def test_duplicate_morton_codes():
    # Many boxes share identical centroids (hence identical Morton codes),
    # exercising the index-tiebreak path in the shared-prefix computation.
    rng = np.random.default_rng(11)
    centers = rng.integers(0, 4, size=(200, 2)) * 0.25  # coarse grid -> ties
    sizes = rng.uniform(0.05, 0.3, size=(200, 2))
    boxes = np.stack([centers, centers + sizes], axis=1)
    got = find_intersections(boxes)
    expected = brute_force(boxes)
    np.testing.assert_array_equal(got, expected)


def test_nan_boxes_do_not_crash():
    rng = np.random.default_rng(13)
    boxes = rand_boxes_2d(100, rng)
    boxes[::7] = np.nan
    got = find_intersections(boxes)
    assert got.ndim == 2 and got.shape[1] == 2


def test_all_overlapping():
    rng = np.random.default_rng(0)
    # identical boxes -> all pairs intersect
    boxes = np.tile(np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32), (100, 1))
    got = find_intersections(boxes)
    assert got.shape == (100 * 99 // 2, 2)


def test_flat_layouts():
    rng = np.random.default_rng(7)
    nested2 = rand_boxes_2d(200, rng)
    flat2 = np.concatenate([nested2[:, 0], nested2[:, 1]], axis=1)
    np.testing.assert_array_equal(
        find_intersections(nested2), find_intersections(flat2)
    )

    nested3 = rand_boxes_3d(200, rng)
    flat3 = np.concatenate([nested3[:, 0], nested3[:, 1]], axis=1)
    np.testing.assert_array_equal(
        find_intersections(nested3), find_intersections(flat3)
    )


def test_float64_input():
    rng = np.random.default_rng(3)
    boxes = rand_boxes_2d(100, rng).astype(np.float64)
    np.testing.assert_array_equal(
        find_intersections(boxes), find_intersections(boxes.astype(np.float32))
    )


def test_sorted_deterministic():
    rng = np.random.default_rng(9)
    boxes = rand_boxes_3d(300, rng)
    got = find_intersections(boxes)
    assert np.all(got[:, 0] < got[:, 1])
    # lexicographically sorted
    assert np.all(got[:-1, 0] <= got[1:, 0])


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        find_intersections(np.zeros((5, 5), dtype=np.float32))
