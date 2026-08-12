# 任务 3.3 学习指南：Prefill——让第一个 token 蹦出来

## 任务目标

`infer(prompt)` 返回的 token 与 HF 生成的**第一个新 token**一致。此步可以把 KV cache 当"恰好只写一次的 buffer"，重点是把 28 层数据流打通。

---

## 数据流（s = prompt 长度，hs=1536，dh=128，di=8960）

C++ 侧全部用 `Tensor::create` 建中间张量（shared_ptr 自动回收）：

```
0.  ids = i64 tensor [s] ← load(prompt 数组)
    hidden = embedding 输出 [s, hs]        （out 2-D，index 1-D i64，weight=in_embed）
每层 i ∈ [0, 28)：
1.  normed = rms_norm(hidden, attn_norm_w[i], eps)          # eps=1e-6！
2.  q = linear(normed, q_w[i], q_b[i]) [s,1536] → view [s,12,128]
    k = linear(normed, k_w[i], k_b[i]) [s,256]  → view [s,2,128]
    v = linear(normed, v_w[i], v_b[i]) [s,256]  → view [s,2,128]
3.  pos_ids = i64 tensor [s] ← [past, past+1, ..., past+s-1]
    q2 = rope(q, pos_ids, θ)；k2 = rope(k, pos_ids, θ)       # θ=10000
4.  kcache[i].slice(0, past, past+s) ← k2   （memcpy D2D）
    vcache[i].slice(0, past, past+s) ← v
5.  attn = self_attention(q2,
                          kcache[i].slice(0, 0, past+s),
                          vcache[i].slice(0, 0, past+s),
                          scale = 1/√128)                   # [s,12,128]
    attn2 = attn.view([s, hs])
6.  tmp = linear(attn2, o_w[i])   —— 无 bias，传 nullptr
    hidden = add(hidden, tmp)                                # 残差
7.  normed = rms_norm(hidden, mlp_norm_w[i], eps)
8.  g = linear(normed, gate_w[i]); u = linear(normed, up_w[i])   # [s, 8960]
    a = swiglu(g, u)                # gate 过 sigmoid，up 做乘数，别反
    d = linear(a, down_w[i])                                   # [s, 1536]
    hidden = add(hidden, d)
收尾：
9.  last = hidden.slice(0, s-1, s)                           # [1, 1536]
10. nl = rms_norm(last, out_norm_w, eps)
11. logits = linear(nl, out_embed) [1, voc] → view [voc]
12. argmax(logits) → max_idx i64[1] → 读回 int64 → past_len += s → return
```

**第 9 步的顺序**：先 slice 再 norm——rms_norm 逐行独立，对最后一行单独 norm 与全量 norm 后取末行完全等价，但省掉 (s−1) 行的 norm 和 (s−1) 倍的 lm_head 矩阵乘（decode 快的前提之一）。

---

## 三个张量操作的连续性前提（作业#1 的判断力在这里变现）

- linear / self_attention 输出都是新建连续张量 → `view` 合法（view 要求连续）；
- `slice(0, ...)` 沿 dim 0 → 结果连续（strides 不变、offset 前移），喂给要求连续的算子合法；
- 不要对 view 成 [s,12,128] 的张量再 slice 维 1/2 后喂算子——非连续，CHECK 会挂。

---

## 写 cache 的方式

目标是连续内存，两条路：

- CPU 直接 `std::memcpy(dst->data(), src->data(), s×nkvh×dh×elementSize)`；
- 面向作业#4 的通用写法：`core::context().runtime().api()->memcpy_sync(..., LLAISYS_MEMCPY_D2D)`（用法照抄 `Tensor::load`，tensor.cpp:293-302）。**建议直接写后者**，作业#4 少改一处。

---

## 读回 argmax 结果

CPU 设备：`max_idx->data()` 直接 `reinterpret_cast<int64_t*>` 读；通用写法用 runtime api 的 memcpy_sync D2H。注意 argmax 要求 vals 是 1-D——logits 先 view 成 [voc]。

---

## 对拍第一个 token

HF 打印的 tokens **包含 prompt**，第一个新 token 是 `tokens[len(prompt_ids)]`。你的 infer 返回值等于它 → M2 通过。不一致 → 停在这里，进 3.5 的对拍流程，别急着写 decode。

---

## 阅读路线（约 20 分钟）

1. `notes/assignment-3-study-guide.md` "单层 forward 数据流"节。
2. `src/ops/` 下每个算子的 op.cpp——每个调用的 CHECK 前置条件。
3. `src/tensor/tensor.cpp` 的 view/slice——连续性前提的实现。
4. runtime api 里 memcpy_sync 的签名（src/device/ 或 include/llaisys/runtime.h）。

**思考问题：**

- 为什么"先 slice 末行再 rms_norm"等价？（逐行独立性）
- prefill 时 qlen == kvlen，self_attention 的 causal 条件 `j ≤ kvlen−qlen+i` 退化成什么？
- embedding 的 index 为什么必须 i64？python 的 list[int] 怎么变成 i64 张量？（提示：ctypes 数组 + load）
- add 的 c 和 a 是同一个张量（in-place 残差）安全吗？看 add_cpu 的循环回答。

---

## 自测清单

- [ ] 默写一层里 12 个权重各自出现在哪一步。
- [ ] 解释 o_proj 的 linear 为什么 bias 传 nullptr。
- [ ] 指出 infer 里 `past_len_ += ntoken` 的确切位置并说明为什么在那里。

---

## 实现提示

- 先写 prefill-only 版本（past=0，attention 直接用 k2/v2 而不读 cache），第一个 token 对了再接通 cache 读取（3.4）——但写 cache 的代码先留着。
- 每层抽成 `layer_forward(i, hidden, pos_ids)`；中间张量循环内 create，别在循环外复用。
- 加个 debug 开关（成员变量或环境变量）：每层末打印 hidden 的 debug()，3.5 对拍全靠它。

---

## 反面教材

- qkv 忘 bias；o_proj 加 bias。
- eps 用 1e-5（config 是 **1e-6**）。
- rope 的 pos_ids 从 0 开始——prefill 恰好没错，decode 全错（潜伏期最长的坑）。
- view 维序写反：[s,12,128] 写成 [s,128,12]。
- 残差加错对象：add(normed, tmp) 而不是 add(hidden, tmp)。
- 中间张量建成 f32——CHECK_SAME_DTYPE 挂。

---

## 验证步骤

python 侧临时脚本：`m.generate(prompt_ids, max_new_tokens=1)` 返回序列的第二个元素 == HF tokens 的对应位置；或直接跑 test_infer.py 看第一个分叉点。
