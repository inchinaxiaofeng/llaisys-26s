# 任务 3.4 学习指南：KV-Cache 与 decode 循环

## 任务目标

generate 跑满 128 步、输出通顺文本；decode 每步只算 1 个 token（qlen=1），耗时**不随**序列变长而增长。

---

## cache 布局与连续性证明

`kcache[i]`：`[maxseq, nkvh, dh]` 连续分配。

- 写：`slice(0, past, past+s)` → shape [s,2,128]、strides 不变、offset 前移 → **连续** ✓
- 读：`slice(0, 0, past+s)` → [past+s,2,128] 连续 ✓（从 0 开始的 dim0 前缀，等价于一块更小的连续张量）

self_attention 要求 k/v 连续——恰好满足。这就是 1.5 思考题"slice 哪一维保连续"的实战用途。

---

## 状态机

```
create:        past_len_ = 0
infer(ids, n): pos_ids = [past, ..., past+n-1]
               forward（用旧 past 写 cache[past : past+n]）
               past_len_ += n
               return next_token
```

更新时机：forward 全部用完 past 之后、return 之前。错位一位，decode 的 rope 位置和 causal 全崩。

---

## decode 形态（n=1）

- ids [1]；q [1,12,128]；写 cache[past:past+1]；
- self_attention：qlen=1、kvlen=past+1 → causal 条件 `j ≤ past+1−1+0 = past` → **全部可见**，自动正确；
- pos_ids = [past]（一个元素的 i64 张量）——3.3 埋的 pos 坑在这里现形。

---

## generate 循环（python）

```python
outputs = list(inputs)
next_id = infer(outputs)              # prefill 整段 prompt
outputs.append(next_id)
while next_id != end_token and 已生成数 < max_new_tokens:
    next_id = infer([next_id])        # decode：只喂新 token
    outputs.append(next_id)
return outputs
```

两个关键：

1. **返回值包含 prompt**——test_infer.py:148 的 `llaisys_tokens == tokens` 里，HF 的 tokens 是 `outputs[0].tolist()`（HF generate 返回输入+输出全序列）。漏掉 prompt 直接全挂。
2. **先 append 再判停止**——HF 生成的 eos 也在序列里；先判后加会少一个 token。

`max_new_tokens=None` 给个兜底（如 128）或直到 eos。

---

## top_k / top_p / temperature 怎么办

`--test` 固定 (1, 1.0, 1.0) → 纯 argmax，C API 只回 int64 正好够。完整采样需要扩展 API（比如 infer 返回 logits，python 侧采样；README 允许改代码），但**不是判定点**。建议：`top_k==1` 走 argmax 主路，其余值 `raise NotImplementedError` 或忽略并注释——别为不考的功能加复杂度。

---

## maxseq 的选择

内存 = 28 层 × 2(k,v) × maxseq × 2 头 × 128 × 2 B = **maxseq × 28 KB**。
4096 → 114 MB；config 的 131072 → 3.6 GB（能跑但浪费）。
建议 4096～8192，并在 infer 入口检查 `past + n ≤ maxseq`，越界**报错**而不是静默写穿。

---

## 性能自检

decode 一步的线性层 FLOPs ≈ 2×1536×(1536×3 + 256×2 + 8960×3 + 1536) + lm_head 约 0.47 GFLOP——自己算一遍。CPU 单线程应跑出**秒级/token**；分钟级 = 大概率没写 cache 在全量重算。

---

## 阅读路线（约 15 分钟）

1. `src/ops/self_attention/op.cpp` 的 kvlen ≥ qlen 检查。
2. `src/tensor/tensor.cpp` 的 slice——offset 单位与 dim0 保连续。
3. `test/test_infer.py:62-79`——llaisys_infer 怎么调 generate、拿什么比较。
4. 总指南"关键设计决策"节。

**思考问题：**

- generate 被调第二次会怎样？（past_len 不归零，接续旧状态）需要 reset API 吗？test 需要吗？
- 写 cache 的 memcpy 源是 rope 输出张量——它的生命周期要保证到什么时候？（memcpy_sync 是同步的，所以？）
- decode 时 q 的 view [1,1536]→[1,12,128] 合法吗？
- 为什么 eos 要先 append 再停？

---

## 自测清单

- [ ] 画出 prefill 后、第一次 decode 后 cache 有效行的示意图。
- [ ] 解释 qlen=1 时 causal 为何自动正确。
- [ ] 默写 generate 的三种退出路径（eos / max_new / None 兜底）。

---

## 实现提示

- M3a：`max_new_tokens=3`，前几个 token 和 HF 逐一对——早期分叉最好查；
- M3b：128 步全跑，肉眼读通顺度；
- 打印每步 token id + decode 出的文本片段，分叉瞬间可见。

---

## 反面教材

- 每次 decode 重新 prefill 全序列（O(n²)，README 点名）。
- 读 cache 读成 `[0, maxseq)` 而不是 `[0, past+s)`。
- past_len 放在 python 侧维护（C API 签名决定状态在模型里）。
- maxseq 越界不检查。
- generate 返回只有新 token（漏 prompt）或少 append eos。

---

## 验证步骤

`python test/test_infer.py --model /root/Model --test` → 128 个 token 全等 → `Test passed!`
