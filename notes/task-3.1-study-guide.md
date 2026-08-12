# 任务 3.1 学习指南：模型契约与四层骨架

## 任务目标

搭通 Python → ctypes → C API → C++ 的完整链路：`Qwen2("/root/Model")` 能创建 C++ 模型对象、读出正确的 meta、拿到 weights 结构体。**本步不算任何 forward**，infer 可以先 `TO_BE_IMPLEMENTED()`——但每一层代码都必须真实存在并编译链接通过。

---

## 分层全景

作业#1/#2 已经建好的链（以 tensor 为例）：

```
python/llaisys/tensor.py        用户 API（Tensor 类）
python/llaisys/libllaisys/*.py  ctypes 包装（argtypes/restype + load_xxx 注册）
src/llaisys/*.cc                C API（__C 块，x->tensor 解包）
src/tensor|ops|.../*.cpp        C++ 实现（静态库，链进 libllaisys.so）
```

模型照搬同一套，四个新文件：

| 层 | 文件 | 内容 |
|---|---|---|
| Python 前端 | python/llaisys/models/qwen2.py | Qwen2 类（已给空壳） |
| ctypes | python/llaisys/libllaisys/qwen2.py（新建） | 结构体镜像 + load_qwen2(lib) |
| C API | src/llaisys/models.cc（新建） | 4 个 __export 函数 |
| C++ | src/models/qwen2.hpp/.cpp（新建） | Qwen2Model 类 |

---

## 契约解读（include/llaisys/models/qwen2.h）

- `LlaisysQwen2Meta`：11 个字段。dtype 枚举 + 8 个 size_t 形状参数 + 2 个 float + 1 个 int64。**顺序和对齐在 ctypes 镜像里必须一字不差**。
- `LlaisysQwen2Weights`：3 个单张量 + 11 个**指针数组**（`llaisysTensor_t *`，每层一个元素）。含义：模型在 create 时就分配好全部权重张量，weights() 把句柄交给 python 灌数据。
- `struct LlaisysQwen2Model;` 只有前置声明——**opaque struct**。C 头不知道模型里有什么，C++ 类随便设计。这是 C/C++ 边界的标准做法：C 侧只摸指针。
- `create(meta, device, device_ids, ndevice)`：ndevice/device_ids 为作业#4 的多卡预留，本作业 ndevice=1、ids=[0]。
- `infer(model, token_ids, ntoken) -> int64`：前向一次、采样一个、返回一个。状态（past_len）在模型内部。

---

## C++ 模型类设计要点

成员建议（src/models/qwen2.hpp）：

- meta 副本（C++ 侧可直接 include 这个 C 头——它是纯 C 兼容的）；
- 单张量：`tensor_t in_embed_, out_embed_, out_norm_w_;`
- 逐层：`std::vector<tensor_t> attn_norm_w_, attn_q_w_, ...`（11 组）；
- KV cache：`std::vector<tensor_t> kcache_, vcache_;`（每层 [maxseq, nkvh, dh]）；
- `size_t past_len_ = 0;`

**生命周期陷阱**：weights() 返回的 `llaisysTensor_t` 是 `LlaisysTensor*`（src/llaisys/llaisys_tensor.hpp:7-9，一个包着 tensor_t 的结构体）。这些包装对象必须**由模型持有**（成员数组），weights() 只装配指针——返回指向栈上临时变量的指针是本任务最经典的崩法。

装配模式：

```
// 模型成员：
LlaisysQwen2Weights weights_;                       // 最终返回的结构体
std::vector<std::unique_ptr<LlaisysTensor>> wrap_;  // 所有包装对象的宿主
std::vector<llaisysTensor_t> attn_q_w_ptrs_;        // 每组一个指针数组，共 11 个
// create 末尾：weights_.attn_q_w = attn_q_w_ptrs_.data(); ...（11 组都接上）
```

---

## ctypes 镜像

```
class LlaisysQwen2Meta(Structure):
    _fields_ = [("dtype", llaisysDataType_t),   # 本质是 c_int 枚举
                ("nlayer", c_size_t), ("hs", c_size_t), ("nh", c_size_t),
                ("nkvh", c_size_t), ("dh", c_size_t), ("di", c_size_t),
                ("maxseq", c_size_t), ("voc", c_size_t),
                ("epsilon", c_float), ("theta", c_float),
                ("end_token", c_int64)]

class LlaisysQwen2Weights(Structure):
    _fields_ = [("in_embed", llaisysTensor_t), ("out_embed", llaisysTensor_t),
                ("out_norm_w", llaisysTensor_t),
                ("attn_q_w", POINTER(llaisysTensor_t)), ...]  # 数组→POINTER，w.attn_q_w[i] 索引
```

函数签名：create → `(POINTER(Meta), c_int, POINTER(c_int), c_int)` 返回 `c_void_p`（opaque 模型指针）；weights → 收模型指针返回 `POINTER(Weights)`；infer → `(c_void_p, POINTER(c_int64), c_size_t)` 返回 `c_int64`。写完 `load_qwen2(lib)` 别忘在 `libllaisys/__init__.py:37-40` 后面注册。

---

## xmake 注册

仿照 llaisys-ops（xmake.lua:82-94）：新 target `llaisys-models`，static，`add_files("src/models/*.cpp")`，`add_deps("llaisys-ops")`（间接链到 tensor/core/device），cxx17、fPIC、`on_install` 空函数照抄；然后在 `target("llaisys")` 里 `add_deps("llaisys-models")`。models.cc 不用单独登记——`src/llaisys/*.cc` 已被 glob 收编。

---

## 阅读路线（约 20 分钟）

1. `include/llaisys/models/qwen2.h` — 契约全文。
2. `src/llaisys/llaisys_tensor.hpp` + `tensor.cc` — opaque 包装与解包的样子。
3. `python/llaisys/libllaisys/llaisys_types.py` — 枚举到 c_int 的映射。
4. `python/llaisys/libllaisys/ops.py` + `__init__.py` — load_xxx 注册模式。
5. `xmake.lua` — target 依赖图。
6. `src/tensor/tensor.hpp` — C++ 侧你能调的全部 API。

**思考问题：**

- 为什么 weights 的逐层字段是 `llaisysTensor_t *` 而不是定长数组？ctypes 侧为什么对应 POINTER 而不是 `llaisysTensor_t * 28`？
- create 时为什么就要把权重张量分配好？（提示：weights() 的用途）
- 如果把 LlaisysTensor 包装对象放在 weights() 的局部变量里，python 第一次 tensorLoad 时会发生什么？
- meta 的 maxseq 在 create 时用来干什么？

---

## 自测清单

- [ ] 画出四层调用链，每层指出一个本任务要写的文件。
- [ ] 默写 meta 的 ctypes `_fields_`（类型 + 顺序）。
- [ ] 一句话解释 opaque struct 解决了什么问题。

---

## 实现提示

顺序：qwen2.hpp（类声明）→ qwen2.cpp（构造分配张量 + infer 先 TO_BE_IMPLEMENTED）→ models.cc（四个函数，照 ops.cc 的解包模式）→ xmake → libllaisys/qwen2.py → __init__ 注册 → models/qwen2.py 先只读 config、填 meta、create、打印。每步编译通过再下一步。

---

## 反面教材

- `_fields_` 顺序和 C 头不一致（不报错，数据错位，最难查）。
- weights() 返回局部变量指针。
- 忘加 fPIC——静态库链进 .so 时链接错误。
- C 头文件里 include C++ 头（破坏 C 兼容性）。

---

## 验证步骤

`xmake && xmake install` 通过；`nm -D lib/libllaisys.so | grep Qwen2` 看到四个导出符号；python 侧 `Qwen2("/root/Model")`（权重加载先留空）不崩且能打印 meta。
