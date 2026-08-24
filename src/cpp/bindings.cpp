#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>

#include <array>
#include <vector>

#include "bvh.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Convert a (n, 2*dim) float array into a vector of AABBs.
// `cast_to_float` handles float64 input by letting numpy copy as float32.
template <int Dim>
std::vector<lbvh2::AABB<Dim>> from_flat(nb::ndarray<float, nb::shape<-1, 2 * Dim>,
                                       nb::c_contig, nb::device::cpu> arr) {
    size_t n = arr.shape(0);
    std::vector<lbvh2::AABB<Dim>> out(n);
    const float* p = arr.data();
    for (size_t i = 0; i < n; ++i) {
        lbvh2::AABB<Dim> box;
        for (int d = 0; d < Dim; ++d) {
            box.min[d] = p[i * 2 * Dim + d];
            box.max[d] = p[i * 2 * Dim + Dim + d];
        }
        out[i] = box;
    }
    return out;
}

// Convert a (n, 2, dim) float array into a vector of AABBs.
template <int Dim>
std::vector<lbvh2::AABB<Dim>> from_nested(nb::ndarray<float, nb::shape<-1, 2, Dim>,
                                         nb::c_contig, nb::device::cpu> arr) {
    size_t n = arr.shape(0);
    std::vector<lbvh2::AABB<Dim>> out(n);
    const float* p = arr.data();
    for (size_t i = 0; i < n; ++i) {
        lbvh2::AABB<Dim> box;
        for (int d = 0; d < Dim; ++d) {
            box.min[d] = p[i * 2 * Dim + d];
            box.max[d] = p[i * 2 * Dim + Dim + d];
        }
        out[i] = box;
    }
    return out;
}

// Find all intersecting pairs, returning an (n, 2) int32 array of original ids.
// Pairs are reported with i < j. `sort_pairs=true` sorts them lexicographically
// (slower, but deterministic / easier to test); `sort_pairs=false` emits them
// in BVH traversal order (matches the reference `lbvh` PyPI package).
template <int Dim>
nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 2>> find_pairs(
    const std::vector<lbvh2::AABB<Dim>>& boxes, bool sort_pairs) {
    lbvh2::BVH<Dim> bvh;
    bvh.build(boxes);

    std::vector<std::array<int32_t, 2>> pairs;
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
        bvh.query(boxes[i], [&](int j) {
            if (i < j) {
                pairs.push_back({i, j});
            }
        });
    }
    if (sort_pairs) {
        std::sort(pairs.begin(), pairs.end());
    }

    int32_t* data = new int32_t[pairs.size() * 2];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 2>> out(
        data, {pairs.size(), 2}, owner);
    for (size_t i = 0; i < pairs.size(); ++i) {
        out(i, 0) = pairs[i][0];
        out(i, 1) = pairs[i][1];
    }
    return out;
}

}  // namespace

NB_MODULE(_lbvh2, m) {
    m.doc() = "LBVH: fast axis-aligned bounding box intersection (self-hosted reimplementation)";

    // sort_pairs=False by default: matches the reference `lbvh` package's
    // BVH traversal order. Pass sort_pairs=True for lexicographically
    // sorted output (deterministic, easier to test, ~30% slower at n=50k).
    m.def("find_intersections",
          [](nb::ndarray<float, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu> a,
             bool sort_pairs) {
              return find_pairs<2>(from_flat<2>(a), sort_pairs);
          },
          nb::arg("boxes"), nb::arg("sort_pairs") = false);

    m.def("find_intersections",
          [](nb::ndarray<float, nb::shape<-1, 6>, nb::c_contig, nb::device::cpu> a,
             bool sort_pairs) {
              return find_pairs<3>(from_flat<3>(a), sort_pairs);
          },
          nb::arg("boxes"), nb::arg("sort_pairs") = false);

    m.def("find_intersections",
          [](nb::ndarray<float, nb::shape<-1, 2, 2>, nb::c_contig, nb::device::cpu> a,
             bool sort_pairs) {
              return find_pairs<2>(from_nested<2>(a), sort_pairs);
          },
          nb::arg("boxes"), nb::arg("sort_pairs") = false);

    m.def("find_intersections",
          [](nb::ndarray<float, nb::shape<-1, 2, 3>, nb::c_contig, nb::device::cpu> a,
             bool sort_pairs) {
              return find_pairs<3>(from_nested<3>(a), sort_pairs);
          },
          nb::arg("boxes"), nb::arg("sort_pairs") = false);
}
