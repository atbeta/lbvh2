#pragma once

#include <cstdint>

namespace lbvh2 {
namespace morton {

// Spread the bits of a 21-bit integer so that bit i lands at bit 3i.
// Standard "part1by2" bit interleaving; used to build a 3D Morton code
// from three 21-bit coordinates.
inline uint64_t expand3(uint64_t a) {
    a &= 0x1FFFFFull;
    a = (a | (a << 32)) & 0x001F00000000FFFFull;
    a = (a | (a << 16)) & 0x001F0000FF0000FFull;
    a = (a | (a << 8))  & 0x100F00F00F00F00Full;
    a = (a | (a << 4))  & 0x10C30C30C30C30C3ull;
    a = (a | (a << 2))  & 0x1249249249249249ull;
    return a;
}

// Spread the bits of a 32-bit integer so that bit i lands at bit 2i.
// Standard "part1by1"; used to build a 2D Morton code.
inline uint64_t expand2(uint64_t a) {
    a &= 0xFFFFFFFFull;
    a = (a | (a << 16)) & 0x0000FFFF0000FFFFull;
    a = (a | (a << 8))  & 0x00FF00FF00FF00FFull;
    a = (a | (a << 4))  & 0x0F0F0F0F0F0F0F0Full;
    a = (a | (a << 2))  & 0x3333333333333333ull;
    a = (a | (a << 1))  & 0x5555555555555555ull;
    return a;
}

// 3D Morton code: 21 bits per axis -> 63-bit code.
inline uint64_t encode3(uint64_t x, uint64_t y, uint64_t z) {
    return (expand3(x) << 2) | (expand3(y) << 1) | expand3(z);
}

// 2D Morton code: 32 bits per axis -> 64-bit code.
inline uint64_t encode2(uint64_t x, uint64_t y) {
    return (expand2(x) << 1) | expand2(y);
}

}  // namespace morton
}  // namespace lbvh2
