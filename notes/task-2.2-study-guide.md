# 任务 2.2 学习指南：实现 `embedding` 算子

## 任务目标

```cpp
void embedding(tensor_t out, tensor_t index, tensor_t weight);
```

语义：**按行收集（gather）**——`index` 是 1-D i64 张量（n 个行号），`weight` 是 2-D 词表矩阵 (vocab, d)，把 `weight` 中编号为 `index[i]` 的行拷到 `out` 的第 i 行。`out` 形状 (n, d)。测试：`test/ops/embedding.py`，idx (1,)/(50,) × embd (2,3)/(512,4096) × f32/f16/bf16。

这是 Assignment #2 里**最不像计算的算子**：零算术，纯搬运。它在 Assignment #3 里是 token → 向量的入口。

---

## 核心：这个算子和 argmax 的三点本质差异

1. **dtype 不参与语义**。argmax 要比较大小，所以离不开 `cast<float>`；embedding 只是把一段字节原样挪窝——f32 的 4 字节和 f16 的 2 字节没有区别。dtype 的唯一作用是算 `elementSize()`。这意味着 kernel **可以不要模板**：按 `std::byte*` + `std::memcpy` 走。（照抄 switch+模板也能过，但要想清楚为什么这里不需要。）
2. **strict 且没有 OR 兜底**。断言是 `check_equal(out_, out, strict=True)`——纯拷贝天然位级相等，没有任何借口。
3. **测试本身有坑**（见下）。

---

## 阅读路线（约 15 分钟）

1. `test/ops/embedding.py` 全文 — 四个点：11 行 torch 参照 `out[:] = embd[idx]`（fancy indexing）；23 行 `random_int_tensor(..., high=embd_shape[0])` 保证行号合法；24 行 `out_` 用 **random_tensor** 初始化；**28 行的 `check_equal` 前面少了什么？**
2. 你的 `src/ops/argmax/op.cpp` — 校验清单这次怎么列？
3. `src/ops/argmax/cpu/argmax_cpu.cpp` — 对照着决定 kernel 形态：需要模板吗？
4. `src/ops/embedding/op.cpp:4-6` — stub 本体。

**思考问题：**

- 28 行 `check_equal(out_, out, strict=True)` 和 argmax 测试的断言写法差了一个关键字，是什么？后果是什么？（提示：回头看 check_equal 内部，只有哪两条是 assert 的？值 mismatch 时会怎样？）跑完测试你**凭什么**确认自己过了？

- `out_` 为什么用 random_tensor 而不是 zero_tensor 初始化？随机残留能抓到什么 zero 初始化抓不到的 bug？

- kernel 不用模板、直接按字节 memcpy 的写法，省掉了什么、放弃了什么？什么算子**必须**有模板？（判断标准：dtype 是否参与语义）

- 校验清单列出来：三个张量各自的 ndim、dtype、shape 关系、连续性，至少 6 条。

- `index` 里的行号可能 ≥ `weight->shape()[0]` 吗？测试保证了吗？要不要在 kernel 里逐个查？查的代价和不查的风险各是什么？

- 逐行 memcpy 和逐元素双重循环拷贝，结果一样，差别在哪？

---

## 自测清单（能回答才算读懂）

- [ ] 五层调用链默写（这次应该 30 秒内写完，包括每一层的"已写好/要写"状态）。
- [ ] embedding 的 op.cpp 校验列表，逐条说出依据（哪条来自 README 契约、哪条来自测试、哪条来自防御）。
- [ ] 循环次数该用谁的 numel？行长度是 `weight->shape()[?]`？写错成对方的 numel 会怎样？
- [ ] 如果把 `index` 按 f32 读会发生什么？i64 的 8 字节读成两个 float，值是多少数量级？
- [ ] 为什么这个测试缺了 assert 也能抓 shape/dtype 错误，却抓不住值错误？

---

## 实现提示

**`src/ops/embedding/op.cpp`**（约 15 行，照 argmax 骨架）：

1. `CHECK_SAME_DEVICE(out, index, weight)`；
2. `CHECK_ARGUMENT(index->ndim() == 1 && index->dtype() == LLAISYS_DTYPE_I64, ...)`；
3. `CHECK_ARGUMENT(weight->ndim() == 2, ...)`；
4. `CHECK_ARGUMENT(out->ndim() == 2 && out->shape()[0] == index->numel() && out->shape()[1] == weight->shape()[1], ...)`；
5. `CHECK_ARGUMENT(out->dtype() == weight->dtype(), ...)`；
6. `ASSERT` 三者 contiguous；
7. dispatch 照抄 argmax（CPU fast-path + setDevice + switch + NVIDIA 骨架）。

**`src/ops/embedding/cpu/embedding_cpu.{hpp,cpp}`**（约 20 行）：推荐按字节写——入口收 `std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_index, size_t row_numel`；`utils::dsize(type)` 算行字节数 `row_bytes`；循环 n 次：`idx = reinterpret_cast<const int64_t *>(index)[i]`，然后 `std::memcpy(out + i*row_bytes, weight + idx*row_bytes, row_bytes)`。

**反面教材（不要这么写）：**

- 逐元素双重循环拷贝——能过，但说明没看出"每一行是一段连续字节"；
- 对 f16/bf16 走 `cast<float>` 再转回——纯拷贝零转换，往返是白送的精度风险；
- 循环次数用 `out->numel()` 或 `weight->numel()`——是 `index->numel()`（行数），别和每行长度搞混；
- 漏掉 `out->shape()[1] == weight->shape()[1]`——shape 不匹配时 memcpy 直接越界；
- 看到 `Test passed!` 就收工——这次它可能是假的，先确认输出里没有 "LLAISYS result: / Torch answer:" 的 mismatch 打印。

---

## 验证步骤

```bash
xmake --root && xmake install --root
cd test && python ops/embedding.py
```

**预期现象**：六组全过且无 mismatch 打印。加 `--profile` 对比 torch。顺手 `python ops/argmax.py && python ops/add.py` 回归。

---

## 学习建议

- 把 argmax 和 embedding 的 kernel 摆在一起看：一个需要模板+cast（有比较语义），一个按字节走（纯搬运）。**"dtype 是否参与语义"是算子实现的第一分类问题。**
- 这个算子暴露了新的测试阅读陷阱：没有 assert 的 check_equal。Assignment #3 调模型时，"测试说过了"和"真的对了"的距离会反复出现——现在先长记性。
- embedding + argmax + 后面的 linear/rms_norm/rope/self_attention/swiglu 凑齐，就是 Qwen2 一层 forward 的全部零件。每写完一个，想想它在模型里的位置。
