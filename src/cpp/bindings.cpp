#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "bvh.hpp"

namespace py = pybind11;

namespace {

// Convert a (n, 2*dim) float array into a vector of AABBs.
template <int Dim>
std::vector<lbvh2::AABB<Dim>> from_flat(py::array_t<float> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 2 * Dim) {
        throw std::runtime_error(
            "expected shape (n, " + std::to_string(2 * Dim) + ") for Dim=" +
            std::to_string(Dim) + ", got (" +
            std::to_string(buf.shape[0]) + ", " +
            std::to_string(buf.shape[1]) + ")");
    }
    size_t n = static_cast<size_t>(buf.shape[0]);
    std::vector<lbvh2::AABB<Dim>> out(n);
    const float* p = static_cast<float*>(buf.ptr);
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
std::vector<lbvh2::AABB<Dim>> from_nested(py::array_t<float> arr) {
    auto buf = arr.request();
    if (buf.ndim != 3 || buf.shape[1] != 2 || buf.shape[2] != Dim) {
        throw std::runtime_error(
            "expected shape (n, 2, " + std::to_string(Dim) + ") for Dim=" +
            std::to_string(Dim) + ", got (" +
            std::to_string(buf.shape[0]) + ", " +
            std::to_string(buf.shape[1]) + ", " +
            std::to_string(buf.shape[2]) + ")");
    }
    size_t n = static_cast<size_t>(buf.shape[0]);
    std::vector<lbvh2::AABB<Dim>> out(n);
    const float* p = static_cast<float*>(buf.ptr);
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
py::array_t<int32_t> find_pairs(
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

    auto out = py::array_t<int32_t>(
        {static_cast<long>(pairs.size()), static_cast<long>(2)});
    auto out_buf = out.request();
    int32_t* p = static_cast<int32_t*>(out_buf.ptr);
    for (size_t i = 0; i < pairs.size(); ++i) {
        p[i * 2 + 0] = pairs[i][0];
        p[i * 2 + 1] = pairs[i][1];
    }
    return out;
}

// Single Python-facing entry: dispatch by shape at runtime. nanobind used
// its type system to pick the right overload; pybind11 can't tell the four
// (py::array_t<float>, bool) overloads apart, so we route them here.
py::array_t<int32_t> dispatch_find_pairs(py::array_t<float> boxes, bool sort_pairs) {
    auto buf = boxes.request();
    const auto shape_str = [&]() {
        std::string s = "(";
        for (ssize_t i = 0; i < buf.ndim; ++i) {
            if (i) s += ", ";
            s += std::to_string(buf.shape[i]);
        }
        s += ")";
        return s;
    }();

    if (buf.ndim == 2 && buf.shape[1] == 4) {
        return find_pairs<2>(from_flat<2>(boxes), sort_pairs);
    }
    if (buf.ndim == 2 && buf.shape[1] == 6) {
        return find_pairs<3>(from_flat<3>(boxes), sort_pairs);
    }
    if (buf.ndim == 3 && buf.shape[1] == 2 && buf.shape[2] == 2) {
        return find_pairs<2>(from_nested<2>(boxes), sort_pairs);
    }
    if (buf.ndim == 3 && buf.shape[1] == 2 && buf.shape[2] == 3) {
        return find_pairs<3>(from_nested<3>(boxes), sort_pairs);
    }
    throw std::runtime_error(
        "unsupported box layout " + shape_str +
        "; expected (*, 4), (*, 2, 2), (*, 6), or (*, 2, 3)");
}

}  // namespace

PYBIND11_MODULE(_lbvh2, m) {
    m.doc() = "LBVH: fast axis-aligned bounding box intersection (self-hosted reimplementation)";

    // sort_pairs=False by default: matches the reference `lbvh` package's
    // BVH traversal order. Pass sort_pairs=True for lexicographically
    // sorted output (deterministic, easier to test, ~30% slower at n=50k).
    m.def("find_intersections", &dispatch_find_pairs,
          py::arg("boxes"), py::arg("sort_pairs") = false);
}