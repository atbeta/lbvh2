#pragma once

#include <array>
#include <algorithm>

namespace lbvh2 {

// Axis-aligned bounding box in `Dim` dimensions, single-precision.
template <int Dim>
struct AABB {
    std::array<float, Dim> min;
    std::array<float, Dim> max;

    AABB() = default;

    // Union of two boxes (the tightest box containing both).
    friend AABB merge(const AABB& a, const AABB& b) {
        AABB out;
        for (int d = 0; d < Dim; ++d) {
            out.min[d] = std::min(a.min[d], b.min[d]);
            out.max[d] = std::max(a.max[d], b.max[d]);
        }
        return out;
    }

    // Overlap test.
    friend bool intersects(const AABB& a, const AABB& b) {
        for (int d = 0; d < Dim; ++d) {
            if (a.min[d] > b.max[d] || a.max[d] < b.min[d]) {
                return false;
            }
        }
        return true;
    }

    // Centroid.
    std::array<float, Dim> center() const {
        std::array<float, Dim> c;
        for (int d = 0; d < Dim; ++d) {
            c[d] = 0.5f * (min[d] + max[d]);
        }
        return c;
    }
};

}  // namespace lbvh2
