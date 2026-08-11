# Assignment #2 实现手册（2.3–2.7）

前置：argmax（2.1）和 embedding（2.2）已实现，分别是"模板+cast"和"纯字节搬运"两个样板。本手册覆盖剩下五个算子：**契约（从测试反推）→ 校验清单 → kernel 决策 → 坑**。

## 通用流程（每个算子走一遍）

1. **读测试推契约**：shape×dtype 矩阵、torch 参照、断言容差。
2. **列校验清单**：每个参数的 ndim / numel / dtype / shape 关系 / 连续性 → op.cpp 的 CHECK 列表。
3. **分层**：op.cpp 只校验+dispatch；cpu/ 只做算法（裸指针 + dtype + 尺寸）。
4. **kernel 四连**：dtype 形态（模板+cast or 按字节）→ 初始化 → 主循环 → 写回。
5. **不动已铺好的层**（唯一例外：linear 的 bias=None，见 2.3）。

**半精度总则**：所有计算/累加在 float 里做，逐元素写回时 `cast<T>`。矩阵乘、归约、softmax 用 float 累加器/缓冲。参考容差：f32 1e-5（rope 1e-4）、f16 1e-3、bf16 1e-2。

**dispatch 骨架**（照 argmax/embedding 抄）：CPU fast-path → setDevice → switch + `#ifdef ENABLE_NVIDIA_API` 的 NVIDIA 占位。

---

## 2.3 linear — `Y = xWᵀ + b`

**契约**（test/ops/linear.py）：
- `x` (m, k)、`w` (n, k)、`out` (m, n)，全 2-D 连续。**w 不转置**：w 的第 j 行是第 j 个输出神经元的权重（PyTorch Linear 的 (out_features, in_features) 布局）。
- `bias` (n,) 可选。测试只测了 `use_bias=True`，但 README 要求支持 None。
- shapes：(2,3)/(2,4)/(3,4) 和 (512,4096) 方阵；x 缩放 0.1、w 缩放 0.01（值域小，float 累加稳过）。

**唯一要动"已铺好层"的地方**：bias=None 时 `python/llaisys/ops.py` 的 `bias.lib_tensor()` 会 AttributeError。改法：Python 层传 `bias.lib_tensor() if bias is not None else None`（ctypes  None→NULL）；`src/llaisys/ops.cc` 的 `llaisysLinear` 改成 `bias ? bias->tensor : llaisys::tensor_t(nullptr)`；C++ 侧 `if (bias)` 判断是否加偏置。

**校验**：同设备；三者 2-D；`w->shape()[1] == x->shape()[1]`；`out->shape() == (x->shape()[0], w->shape()[0])`；bias 非空时 1-D、长 n、同 dtype；全连续。

**kernel**（模板 + switch）：
```
for i in [0,m): for j in [0,n):
    float acc = bias ? cast<float>(bias[j]) : 0f
    for k in [0,K): acc += cast<float>(x[i,k]) * cast<float>(w[j,k])
    out[i,j] = cast<T>(acc)
```
w 按行读是连续访存（cache 友好）——这正是"不转置"布局的工程理由。

**坑**：把 w 当 (k,n) 用（结果 shape 对、值全错）；f16 直接累加（4096 长度累加精度炸）；空指针 bias 直接解引用。

---

## 2.4 rms_norm — 逐行 `y = w·x / sqrt(mean(x²)+eps)`

**契约**（test/ops/rms_norm.py）：`in`/`out` (rows, d) 2-D 连续；`weight` (d,) 1-D；`eps` 是 C API 的 float 参数（测试固定 1e-5）。torch 参照顺序：mean(x²) → +eps → rsqrt → ×x → ×w。

**校验**：同设备；in/out 同形 2-D 同 dtype；weight 1-D 长 d 同 dtype；全连续。

**kernel**（模板 + switch）：每行两遍——第一遍 float 累加 `Σx²`，算 `r = 1/sqrt(Σx²/d + eps)`；第二遍 `out[i] = cast<T>(cast<float>(x[i]) * r * cast<float>(w[i]))`。

**坑**：mean 忘除 d（变成 sum）；eps 加到 sum 上而不是 mean 上；把 w 当标量。

---

## 2.5 rope — 半分割旋转（GPT-NeoX 式）

**契约**（test/ops/rope.py，**以此为准**，不是记忆里的交错式）：
- `in`/`out` (seqlen, nhead, d) 3-D 连续，d 偶数；a = 前 d/2，b = 后 d/2。
- `pos_ids` (seqlen,) i64；θ = 10000（C API float 参数）。
- φ(s,j) = pos_ids[s] / θ^(2j/d)，j ∈ [0, d/2)；
  `out[s,h,j] = a_j·cosφ − b_j·sinφ`；`out[s,h,j+d/2] = b_j·cosφ + a_j·sinφ`。
- shapes：(2,1,4) pos(0,2)；(512,4,4096) pos(512,1024)（pos 不从 0 开始，覆盖 decode 场景）。f32 容差 **1e-4**，比别的算子严。
- in/out 是不同张量，out 每个元素都要写。

**校验**：同设备；in/out 同形 3-D 同 dtype；d % 2 == 0；pos_ids 1-D 长 seqlen、I64；全连续。

**kernel**（模板 + switch）：三重循环 s × h × j；pos 按 `int64_t` 读转 float；`std::powf`/循环算 θ^(-2j/d)（可在 s 循环外预计算 d/2 个频率因子）；sin/cos 用 float；算完 `cast<T>` 写回。

**坑**：写成交错式（x[2j], x[2j+1] 配对）——那是 GPT-J 式，和本测试的 half-split 差一个维度排列；指数里 j 是**半维**下标；pos_ids 当 int32 读。

---

## 2.6 self_attention — 本 Assignment 最重的算子

**契约**（test/ops/self_attention.py）：
- `q` (L, nh, hd)、`k`/`v` (S, nkvh, hd)、`attn_val` (L, nh, hd)，全连续。scale = 1/√hd 由 C API 传入。
- **GQA**：nh 是 nkvh 的整数倍；q 的第 h 头用 kv 的第 `h / (nh/nkvh)` 头（repeat_interleave 语义：kv head j 服务连续的一段 q heads）。
- **causal mask（带偏移）**：q 位置 t 只允许看 kv 位置 `s ≤ t + (S − L)`。S > L 是 decode 形态（S=11, L=5 的测试用例就是）。
- 流程：scores = QKᵀ·scale → mask 位置填 −inf → softmax(逐行) → out = P·V。
- README 说的"concat kvcache"测试不涉及——k/v 直接就是全量，concat 留给 Assignment #3 的模型层。

**校验**：同设备同 dtype；q/out 同形 3-D；k/v 同形；`k->shape()[1]` 整除 `q->shape()[1]`（nkvh | nh…实际是 nh % nkvh == 0）；hd 一致；`S >= L`；全连续。

**kernel**（模板 + switch）：h × t 双重循环，每个 (h,t)：
1. 算 S 个 score 存入 `std::vector<float>` scratch：`s ≤ t+(S−L)` 的算 `Σ_j q·k`（float 累加）× scale，其余填 −inf（或跳过参与后面两步）；
2. softmax：减 max → exp → 归一（值域小本测试不炸，但 max-subtraction 是职业习惯）；
3. `out[h,t,:] = Σ_s p_s · v[s, h_kv, :]`，float 累加，cast<T> 写回。

**坑**：mask 偏移 S−L 漏掉（S==L 时碰巧不错，decode 用例全错）；GQA 映射写成 `h % nkvh`（正确是 `h / (nh/nkvh)`）；f16 里直接做 softmax 累加；scratch 按 S 分配错成 L。

---

## 2.7 swiglu — 逐元素收尾

**契约**（test/ops/swiglu.py）：`out = up ⊙ (gate / (1 + e^(−gate)))`，gate/up/out 同形 2-D 连续。注意 torch 参照：sigmoid 部分在 float 里算完 `.to(dtype)` 再乘 up。shapes：(2,3)、(512,4096)。

**校验**：同设备；三者同形 2-D 同 dtype；全连续。

**kernel**（模板 + switch）：单循环 numel：`g = cast<float>(gate[i])`，`out[i] = cast<T>(cast<float>(up[i]) * g / (1f + expf(-g)))`。别搞反谁过 sigmoid：是 gate 过、up 做乘数。

**坑**：`expf(-g)` 在 g 很负时溢出（本测试值域 [0,1) 不会；生产写法可分支稳定化，此处不必）。

---

## 收尾（Task 2.8）

`test/test_ops.py` 不存在——逐个跑：

```bash
cd test
python ops/add.py && python ops/argmax.py && python ops/embedding.py && \
python ops/linear.py && python ops/rms_norm.py && python ops/rope.py && \
python ops/self_attention.py && python ops/swiglu.py && python test_tensor.py
```

**记住 embedding 测试没有 assert**——盯输出里有没有 mismatch 段。全绿后 commit & push，看 GitHub Actions。
