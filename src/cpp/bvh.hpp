#pragma once

#include <cstdint>
#include <vector>

#include "aabb.hpp"
#include "morton.hpp"

namespace lbvh2 {

// Linear bounding volume hierarchy over axis-aligned boxes.
//
// Construction follows the Karras (2012) linear BVH algorithm:
// leaves are sorted by Morton code of their centroid, then internal
// nodes are derived in a single pass using longest-common-prefix
// binary searches, and finally node boxes are merged bottom-up.
template <int Dim>
struct BVH {
    using code_t = uint64_t;

    int num_leaves = 0;
    std::vector<int> ids;                 // original index of each sorted leaf
    std::vector<AABB<Dim>> boxes;         // [0,n) leaves, [n,2n-1) internal nodes
    std::vector<std::array<int, 2>> children;  // child indices per node

    // Build from a set of leaf boxes. `leaves` is indexed by the caller's
    // original ids; those ids are what queries report back.
    void build(const std::vector<AABB<Dim>>& leaves) {
        const int n = static_cast<int>(leaves.size());
        num_leaves = n;
        boxes.clear();
        children.clear();
        ids.clear();

        if (n == 0) {
            return;
        }

        boxes.assign(2 * n - 1, AABB<Dim>{});
        children.assign(2 * n - 1, {-1, -1});
        ids.assign(n, 0);

        // Global bounds (of the centroids, actually the boxes themselves).
        AABB<Dim> global = leaves[0];
        for (int i = 0; i < n; ++i) {
            global = merge(global, leaves[i]);
        }
        std::array<float, Dim> lo = global.min;
        std::array<float, Dim> span;
        for (int d = 0; d < Dim; ++d) {
            span[d] = global.max[d] - global.min[d];
        }

        // Morton codes of leaf centroids, normalized to [0,1).
        std::vector<std::pair<code_t, int>> code_ids(n);
        for (int i = 0; i < n; ++i) {
            auto c = leaves[i].center();
            code_t code = encode(c, lo, span);
            code_ids[i] = {code, i};
        }

        std::sort(code_ids.begin(), code_ids.end());

        // Leaves in sorted order.
        for (int i = 0; i < n; ++i) {
            ids[i] = code_ids[i].second;
            boxes[i] = leaves[ids[i]];
        }

        if (n == 1) {
            return;
        }

        // Longest common prefix length between two sorted leaves.
        // Codes are the primary key; the original index breaks ties so
        // identical Morton codes still yield a consistent order.
        auto shared_prefix = [&](int a, int b) -> int {
            if (a < 0 || a >= n || b < 0 || b >= n) {
                return -1;
            }
            code_t ca = code_ids[a].first, cb = code_ids[b].first;
            int bits = clz(ca ^ cb);
            if (bits == 64) {
                bits += clz32(code_ids[a].second ^ code_ids[b].second);
            }
            return bits;
        };

        std::vector<int> parent(2 * n - 1, -1);
        std::vector<int> sibling(2 * n - 1, -1);

        // Derive the internal node for each leaf i (0 <= i < n-1).
        for (int i = 0; i < n - 1; ++i) {
            int prefix_prev = shared_prefix(i, i - 1);
            int prefix_next = shared_prefix(i, i + 1);
            int prefix_min = std::min(prefix_prev, prefix_next);

            // Search in the direction of the neighbor with the longer prefix.
            int d = (prefix_next > prefix_prev) ? 1 : -1;

            // Exponential upper bound on the range length. Start at 32 like
            // the reference (the exact start value only affects iteration
            // count, not the final result).
            int lmax = 32;
            while (i + lmax * d >= 0 && i + lmax * d < n &&
                   shared_prefix(i, i + lmax * d) > prefix_min) {
                lmax *= 2;
            }

            // Binary search for the farthest leaf with shared prefix > prefix_min.
            int l = 0;
            for (int t = lmax / 2; t > 0; t >>= 1) {
                int probe = i + (l + t) * d;
                if (probe >= 0 && probe < n &&
                    shared_prefix(i, probe) > prefix_min) {
                    l += t;
                }
            }
            int j = i + l * d;
            int prefix_node = shared_prefix(i, j);

            // Binary search for the split point within [0, l]. The step is
            // halved with rounding-up (t = (t+1)>>1) so that the value t=1
            // is probed exactly once, matching the Karras reference.
            int s = 0;
            int t = l;
            do {
                t = (t + 1) >> 1;
                int probe = i + (s + t) * d;
                if (probe >= 0 && probe < n &&
                    shared_prefix(i, probe) > prefix_node) {
                    s += t;
                }
            } while (t > 1);
            int split = i + s * d + std::min(d, 0);

            int lo_i = std::min(i, j);
            int hi_i = std::max(i, j);

            int left  = (lo_i == split) ? split : split + n;
            int right = (hi_i == split + 1) ? split + 1 : split + 1 + n;

            int node = i + n;
            children[node] = {left, right};
            parent[left] = node;
            parent[right] = node;
            sibling[left] = right;
            sibling[right] = left;
        }

        // Bottom-up box merge, serialized per leaf. `visits` counts how many
        // children of each internal node have reported; when both arrive we
        // fold their boxes into the parent and continue upward.
        std::vector<int> visits(2 * n - 1, 0);
        for (int leaf = 0; leaf < n; ++leaf) {
            int id = leaf;
            AABB<Dim> box = boxes[leaf];
            while (parent[id] != -1) {
                int p = parent[id];
                visits[p]++;
                if (visits[p] == 1) {
                    break;  // sibling hasn't reported yet
                }
                box = merge(box, boxes[sibling[id]]);
                boxes[p] = box;
                id = p;
            }
        }
    }

    // Report the original ids of every leaf whose box intersects `query`.
    // `emit` is invoked once per hit.
    template <typename Fn>
    void query(const AABB<Dim>& query, Fn&& emit) const {
        if (num_leaves == 0) {
            return;
        }
        if (num_leaves == 1) {
            if (intersects(boxes[0], query)) {
                emit(ids[0]);
            }
            return;
        }

        // Fixed capacity: Morton codes leave <= 60 significant bits and the
        // same-code tie-break builds a radix tree over 32-bit index bits, so
        // the tree height stays far below 64 for any int32-indexed input.
        int stack[64];
        int sp = 0;
        stack[sp++] = num_leaves;  // root is internal node num_leaves

        while (sp > 0) {
            int node = stack[--sp];
            auto [l, r] = children[node];

            bool hit_l = intersects(boxes[l], query);
            if (hit_l) {
                if (l < num_leaves) {
                    emit(ids[l]);
                } else {
                    stack[sp++] = l;
                }
            }

            bool hit_r = intersects(boxes[r], query);
            if (hit_r) {
                if (r < num_leaves) {
                    emit(ids[r]);
                } else {
                    stack[sp++] = r;
                }
            }
        }
    }

private:
    static int clz(uint64_t x) {
        return x == 0 ? 64 : __builtin_clzll(x);
    }

    static int clz32(uint32_t x) {
        return x == 0 ? 32 : __builtin_clz(x);
    }

    code_t encode(const std::array<float, Dim>& c,
                  const std::array<float, Dim>& lo,
                  const std::array<float, Dim>& span) const {
        // Normalized coordinate in [0,1], clamped to guard against NaN/Inf
        // and float rounding at the boundary (u==1.0 would overflow the cast).
        auto u = [&](int d) -> float {
            if (!(span[d] > 0.0f)) {
                return 0.0f;
            }
            float v = (c[d] - lo[d]) / span[d];
            if (!(v >= 0.0f)) return 0.0f;    // NaN or below range
            if (v > 1.0f) return 1.0f;
            return v;
        };

        if constexpr (Dim == 2) {
            // Largest float32 strictly below 2^32 (2^32 - 2^8): 2^32 - 1 is not
            // representable in float32 and would round up to 2^32, wrapping
            // through expand2's 32-bit mask to code 0 for centroids at v == 1.0.
            constexpr float scale = 4294967040.0f;
            uint64_t x = static_cast<uint64_t>(u(0) * scale);
            uint64_t y = static_cast<uint64_t>(u(1) * scale);
            return morton::encode2(x, y);
        } else {
            constexpr float scale = 2097151.0f;  // 2^21 - 1
            uint64_t x = static_cast<uint64_t>(u(0) * scale);
            uint64_t y = static_cast<uint64_t>(u(1) * scale);
            uint64_t z = static_cast<uint64_t>(u(2) * scale);
            return morton::encode3(x, y, z);
        }
    }
};

}  // namespace lbvh2
