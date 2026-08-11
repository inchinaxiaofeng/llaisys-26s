# 任务 2.1 学习指南：实现 `argmax` 算子

## 任务目标

实现 Assignment #2 的第一个算子：

```cpp
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
```

语义：找出 1-D 张量 `vals` 的最大值，值写入 `max_val[0]`，下标写入 `max_idx[0]`（i64）。测试：`test/ops/argmax.py`，shapes (4,) / (4096,)，dtypes f32 / f16 / bf16。

这个任务算法本身是一遍扫描，**真正的内容是读懂并复制 add 算子的五层调用链**——2.2 到 2.7 全是同一个骨架，2.1 搭熟了后面只剩数学。

---

## 核心：五层调用链，你要写的只有两环

以 add 为样板，一次算子调用的完整路径：

1. `test/ops/argmax.py:28` — `llaisys.Ops.argmax(max_idx_, max_val_, vals_)`
2. `python/llaisys/ops.py:12-13` — Python 包装类 `Ops`（**已写好**）
3. `python/llaisys/libllaisys/ops.py:8-9` — ctypes 签名（**已写好**）
4. `include/llaisys/ops.h:8` — C API 声明（**已写好**）
5. `src/llaisys/ops.cc:19-21` — C API 解包 `LlaisysTensor` 转发给 C++（**已写好**）
6. `src/ops/argmax/op.cpp:4-6` — C++ 算子入口：校验 + 设备分发 ← **stub，要写**
7. `src/ops/argmax/cpu/argmax_cpu.cpp` — CPU kernel：真正的算法 ← **不存在，要新建**

对照 add：`src/ops/add/op.cpp`（校验 + dispatch）和 `src/ops/add/cpu/add_cpu.{hpp,cpp}`（模板 kernel + dtype switch）。你的 argmax 应该在结构上是它的孪生。

**为什么新建 `cpu/argmax_cpu.cpp` 后 xmake 一行都不用改？** 看 `xmake/cpu.lua:23`：`add_files("../src/ops/*/cpu/*.cpp")` 是 glob，新文件自动收编进 `llaisys-ops-cpu` 静态库。同理 `xmake.lua:93` 的 `src/ops/*/*.cpp` 收编 op.cpp。这套"约定优于配置"是整个 Assignment #2 能批量产算子的前提。

**dtype 处理套路**（add_cpu.cpp:7-33）：模板函数 `argmax_<T>` 写算法，入口函数 switch dtype 分发到 `float` / `fp16_t` / `bf16_t` 三个实例化；半精度通过 `utils::cast<float>`（types.hpp:116-139）转成 float 做计算/比较，写完再 `cast<T>` 回去。`default` 分支走 `EXCEPTION_UNSUPPORTED_DATATYPE`。

---

## 阅读路线（约 25 分钟）

1. `test/ops/argmax.py` 全文 — 重点三处：13 行 torch 侧 `torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))` 的 **out 参数顺序**；30-32 行断言为什么是 **OR**；49-50 行的 shapes × dtypes 矩阵。

参数顺序：Pytorch原生torch.max(input, dim, keepdim)函数的返回值顺序固定为(values, indices)（先返回最大数值，后返回最大数值索引）
当使用out关键字参数绑定输出张量时，传入的元组必须严格匹配PyTorch的返回顺序，即out=(max_val, max_idx)

采用OR：当输入Tensor vals中存在相同的最大值（[3,5,5,2\])时，PyTorch的Torch.max保证返回第一个最大值索引，而并行/向量化规约实现（GPU Block Reduction或特定C++多线程规约）在并发处理时，可能会返回另一个最大的相同值。
OR可以允许在数值规约正确的情况下，容忍并行计算的选择差异。
> ✅ 对。一个小订正：torch.max CPU 版"返回第一个最大值下标"是惯例而非硬保证，OR 的设计意图正是 tie 容忍——和你下面思考题的答案对一下，别自相矛盾。

shapes•dtypes矩阵，这是深度学习算子测试中的笛卡尔积测试覆盖矩阵。
> ✅

2. `src/ops/add/op.cpp:9-34` — op.cpp 层干什么、不干什么。校验宏在 `src/utils/check.hpp`：CHECK_SAME_DEVICE / CHECK_SAME_DTYPE / CHECK_ARGUMENT / ASSERT。

给更上一层调用的算子入口函数，核心是合法性检查和设备分发。
不干：不计算Tensor，不包含硬件特化的实现代码，不管理低层显存/内存申请。

四个防御性编程：
1. CHECK_SAME_DEVICE(tensor1, tensor2);
* 断言两个Tensor处于同一个物理设备（deviceType和deviceId完全一致）
* 仿真出现CPU张量与GPU Tensor直接相加的非法跨设备调用
2. CHECK_SAME_DTYPE(tensor1, tensor2);
* 断言两个Tensor的数据类型相同（都是fp32）
* 若算子未实现隐式类型转换，确保传入的数据类型一致
3. CHECK_ARGUMENT(condition, "error_message");
* 校验算子特定的业务逻辑参数（如a->numel() == out->numel() 或start <= end），若条件不满足则抛出非法参数异常（std::invalid_argument）
4. ASSERT(condition, "error_message");
* 内部程序不变量断言，用于捕获理论上绝不应该发生的底层逻辑错误或指针空等Bug
> ✅ 四条都对。补一刀区分：CHECK_ARGUMENT 抛 `std::invalid_argument`（调用者的锅），ASSERT 抛 `std::runtime_error`（实现的锅）——报错类型本身就是责任划分。

3. `src/ops/add/cpu/add_cpu.cpp:7-33` — 模板 + switch 的 kernel 组织；`if constexpr` 处理半精度的地方。
4. `src/utils/types.hpp:116-139` — `utils::cast` 模板，2.x 全程都要用它。
5. `src/ops/argmax/op.cpp:4-6` — stub 本体。
6. 快速过一遍已铺好的中间层（python/llaisys/ops.py:12-13 → libllaisys/ops.py:8-9 → ops.h:8 → ops.cc:19-21），确认确实不用动。

**思考问题：**

- add 的 op.cpp 里，17-19 行的 `if (cpu) return` 和 switch 里的 `case LLAISYS_DEVICE_CPU` 是什么关系？case 能到达吗？这个"冗余"是留给谁的？

理论上来说不会被执行到，case是通用的多设备网关？
> ✅ 对。说透：`if (cpu) return` 是 fast path，保证**没配 GPU 时 CPU 永远可用**；switch 里的 `case CPU` 确实不可达。整个 setDevice + switch + `#ifdef ENABLE_NVIDIA_API` 骨架是留给 Assignment #4 的——那时 `case LLAISYS_DEVICE_NVIDIA` 才有实现。照抄这个骨架到你自己的 op.cpp，不是死代码，是预留的扩展点。

- 测试断言是 `check_equal(max_val_) or check_equal(max_idx_)` 的 **OR** 关系。这对你的实现放宽了什么、又必须坚持什么？（提示：最大值这个**值**唯一吗？并列最大时**下标**唯一吗？）

这里应该是变得严格了吧？需要确保最大值和最大值下标都一样。
> ❌ 反了。OR 是**放宽**：两个里头对一个就过。"两个都必须一样"那是 AND。拆开看：
> - **必须坚持**：`max_val` strict 相等——最大值作为"值"是唯一的（并列时值相同），严格相等一定达得到；
> - **被放宽**：`max_idx`——并列最大时下标不唯一，你的实现和 torch 选了不同的并列元素也能过。
> 为什么需要这个放宽？f16 只有约 6 万个可表示值，bf16 尾数更短，(4096,) 的随机输入**并列最大几乎必然出现**，tie-break 差异必须容忍。你自己在上面阅读路线里写的"容忍并行选择差异"是对的，到这里别又拧回去。

- `fp16_t` / `bf16_t` 能直接 `a > b` 比大小吗？退一步，按 `uint16_t` 位模式直接比行不行？（提示：IEEE 浮点位序对正数保序，对负数呢？）

如你所说，IEEE浮点位序对正数保序，因此可以进行比大小，负数不确定诶，是不是不行？
> 半对，补全：按 uint16 位模式比，负数**反序**——符号位为 1 的负数，位模式越大实际值越小；且所有负数的位模式（≥0x8000）按无符号比都大于正数。结果是负数全被排到正数前面，错得彻底。
> 更根本的一层：`fp16_t`/`bf16_t` 是 `struct { uint16_t _v; }`，**根本没有 `operator>`**，`a > b` 编译都过不去。所以正路只有一条：`utils::cast<float>` 后按 float 比。

- 找到最大值后写回 `max_val`，用 `cast<T>(best_float)` 还是直接拷 `vals[best]`？哪个更稳？为什么？

完全不知道有什么关系
> 补讲：`cast<T>(cast<float>(x))` 对本项目的 cast 实现是**无损往返**（bf16 是左移 16 位再舍回原值；f16 转上来的值 mantissa 低 13 位为 0，舍回不丢），所以两种写法都能过 strict。但工程上**直接拷 `vals[best]` 更稳**：
> ① 语义上 max_val 本来就是输入集合里的一个元素，原样拷回零风险；
> ② 少一次转换，不依赖 cast 实现的正确性——如果 `_f32_to_f16` 的边界有 bug，往返会把噪声写进结果。
> 养成习惯：能保留原始位模式的，绝不做"转出去再转回来"。

- `llaisysArgmax(max_idx, max_val, vals)` 的参数顺序和 torch 侧 `out=(max_val, max_idx)` 一致吗？搞反了会怎样？

不一致。
> ✅ 对，把后半问也答了：搞反了会把最大值**写进 i64 的 max_idx**、把下标**写进 f16 的 max_val**——dtype 对不上，check_equal 的 dtype 断言先炸。还有个阅读陷阱：测试里 `torch_argmax(max_idx, max_val, vals)` 形参名是 idx 在前，但函数体里 `out=(max_val, max_idx)` 才是真相——读测试盯 out 元组，别盯形参名。

- vals 是空张量（numel == 0）怎么办？测试查吗？你的校验该拦吗？

> （没答，补）：测试**不查**——形状只有 (4,) 和 (4096,)。但该拦：numel==0 时"最大值"无定义，`vals[0]` 初始化直接读越界（UB，不保证报错）。`CHECK_ARGUMENT(vals->numel() > 0, ...)` 一行换一个确定的异常，值。

---

## 自测清单（能回答才算读懂）

- [ ] 从 `llaisys.Ops.argmax(...)` 到 `cpu::argmax` 一共几层？逐层说出文件和行号。
- [ ] op.cpp 层该做哪些校验？（至少数出 5 条，含 dtype 和 numel 的约束）
- [ ] f16 跑 (4096,) 时 strict=True 为什么也能过？（联系 cast 往返是否无损）
- [ ] 并列最大值时你的实现返回第一个还是最后一个？`>` 和 `>=` 的区别？
- [ ] 为什么这个新算子不用改 `xmake.lua` / `xmake/cpu.lua`、不用改 Python、不用改 C API？

---

## 实现提示

**`src/ops/argmax/op.cpp`**（约 15 行）：

1. `CHECK_SAME_DEVICE(max_idx, max_val, vals)`；
2. `CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, ...)`；
3. `CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64 && max_idx->numel() == 1, ...)`；
4. `CHECK_ARGUMENT(max_val->dtype() == vals->dtype() && max_val->numel() == 1, ...)`；
5. `ASSERT` 三者 contiguous（照 add 的惯例）；
6. dispatch：`if (vals->deviceType() == LLAISYS_DEVICE_CPU) return cpu::argmax(...)`，再留 setDevice + switch + `#ifdef ENABLE_NVIDIA_API` 的骨架（照抄 add，Assignment #4 要用）。

**`src/ops/argmax/cpu/argmax_cpu.hpp`**：照 add_cpu.hpp，声明 `void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);`

**`src/ops/argmax/cpu/argmax_cpu.cpp`**（约 30 行）：模板 `argmax_<T>(int64_t *max_idx, T *max_val, const T *vals, size_t numel)` 一遍扫描；比较一律经 `utils::cast<float>`；switch 三分支 + default 走 `EXCEPTION_UNSUPPORTED_DATATYPE`。

**反面教材（不要这么写）：**

- 直接 `vals[i] > best` 比较 `fp16_t`/`bf16_t`——没有比较运算符，编译就过不去；强转 uint16 比位模式——负数全错；
- `max_idx` 按 vals 的 dtype 写——它是 i64，和 vals 的 dtype 无关；
- 把算法直接写进 op.cpp——层放错了：op.cpp 只校验 + dispatch，算法归 cpu/；
- 去改 xmake / Python / ops.h / ops.cc——都已铺好，一行不用动；
- 扫描时初始化 best_val 为 0 或 INT_MIN 之类——首元素初始化才安全（想想全负数的输入）。

---

## 验证步骤

```bash
xmake --root && xmake install --root
cd test && python ops/argmax.py
```

**预期现象**：f32 / f16 / bf16 × {(4,), (4096,)} 六组全过，打印 `Test passed!`。顺手 `python ops/add.py` 确认没碰坏 add；加 `--profile` 可以和 torch 对比耗时。

---

## 学习建议

- 写完对比一下自己的 op.cpp 和 add 的 op.cpp：结构上应该是孪生。如果不是，想想差异是有理由的还是没读懂骨架。
- argmax 的"输出参数"风格（调用者分配好 max_idx/max_val，算子只写不分配）是推理框架的典型设计——零分配、hot path 友好。2.2 起全是这个风格。
- 参数顺序陷阱（C API 的 (idx, val) vs torch 的 out (values, indices)）在 Assignment #2 里会反复出现：linear 的 W 不转置也是同类。每个算子先把两侧语义对齐再动笔。
