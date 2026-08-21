# 从原版 `lbvh` 迁移到 `lbvh2`

本文件说明自研 `lbvh2` 与原版 `lbvh`（samuelpmish/LBVH，PyPI 包名 `lbvh`，
`import LBVH`）的差异，以及如何无缝替换。

## 一句话结论

API 完全对齐：同一个 `find_intersections` 函数、同样的 4 种输入布局、同样的
`(N, 2)` int32 返回。**只需改一行 import**，行为上是原版的超集（多支持
float64、返回有序），不破坏既有调用。

---

## 快速替换

```python
# 原版
import LBVH
pairs = LBVH.find_intersections(boxes)

# 自研版（最小改动：一行 import）
import lbvh2 as LBVH
pairs = LBVH.find_intersections(boxes)

# 或直接导入函数
from lbvh2 import find_intersections
pairs = find_intersections(boxes)
```

若存量代码里 `import LBVH` 遍布、不想逐处改，可在内部建一个纯 Python 别名
转发包（无需发布到 PyPI，也不占用 `lbvh` 这个名字）：

```python
# 内部 shim 包 LBVH/__init__.py
from lbvh2 import find_intersections  # noqa: F401
```

---

## API 对比

| 维度 | 原版 `lbvh` | 自研 `lbvh2` | 影响 |
|---|---|---|---|
| 导入名 | `import LBVH` | `import lbvh2` | 需改 import |
| 函数 | `find_intersections` | `find_intersections` | 一致 |
| 2D flat `(*, 4)` | 支持 | 支持 | 一致 |
| 2D nested `(*, 2, 2)` | 支持 | 支持 | 一致 |
| 3D flat `(*, 6)` | 支持 | 支持 | 一致 |
| 3D nested `(*, 2, 3)` | 支持 | 支持 | 一致 |
| 返回类型 | `(N, 2)` int32 | `(N, 2)` int32 | 一致 |
| `i < j` 保证 | 有 | 有 | 一致 |

---

## 行为差异（自研版是超集）

| 维度 | 原版 | 自研版 | 说明 |
|---|---|---|---|
| 输入 dtype | 仅 `float32`，传 `float64` 直接报错 | `float32`/`float64` 均可，自动转 | 超集，不破坏旧调用 |
| 输出顺序 | 无序（遍历顺序不定） | 按 `(i, j)` 字典序排序 | 严格更强，利于回归测试 |
| 输入数组布局 | 要求连续 | 自动 `ascontiguousarray` | 超集 |
| 空输入 `(0, ...)` | 返回空数组 | 返回空数组 | 一致 |
| 单元素输入 | 正常 | 正常 | 一致 |
| C++ 依赖 | 链接 `fm` 库（AABB 类型） | 零外部 C++ 依赖，AABB 内联 | 更易维护 |
| nanobind | 1.x（源码内嵌，cp38–cp311 多 wheel） | 2.x + abi3，单 wheel 覆盖 3.12+ | 打包矩阵更小 |

### 唯一需要留意的点

**输出顺序**：原版返回未排序的 pair 集合；自研版返回字典序排序后的数组。
若下游只把结果当「集合」用（绝大多数情况），无任何影响；若依赖原版的具体
遍历顺序（极少数），需自行排序适配。除此之外全部向后兼容。

---

## 验证替换正确性

替换后可用暴力对拍快速自检（自研版测试即采用此法）：

```python
import numpy as np
from lbvh2 import find_intersections

def brute_force(boxes):  # O(n^2) 参考实现
    boxes = np.asarray(boxes, dtype=np.float32)
    lo = boxes[:, 0]; hi = boxes[:, 1]
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if np.all(lo[i] <= hi[j]) and np.all(lo[j] <= hi[i]):
                pairs.append([i, j])
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)

rng = np.random.default_rng(0)
a = rng.uniform(-1, 1, (500, 3))
boxes = np.stack([a, a + rng.uniform(0, 0.5, (500, 3))], axis=1)

assert np.array_equal(find_intersections(boxes), brute_force(boxes))
```

---

## 构建 / 发布差异

| 维度 | 原版 | 自研版 |
|---|---|---|
| 构建后端 | scikit-build-core + 内嵌 nanobind 1.x | scikit-build-core + nanobind 2.x |
| wheel | cp38–cp311，多平台多版本矩阵 | abi3 单 wheel 覆盖 3.12+ |
| 源码分发 | 需 `fm` + nanobind FetchContent | 纯自包含，`sdist` 含全部头文件 |
| 内部验证 | — | 无需发布 PyPI，CI 出 wheel 到内部制品库即可 |
