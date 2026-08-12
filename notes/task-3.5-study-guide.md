# 任务 3.5 学习指南：对拍——从第一个分叉点定位 bug

## 任务目标

掌握系统化对拍：任何不一致都在 10 分钟内定位到具体层的具体算子。**先把 3.3/3.4 做完再来。**

---

## 为什么"逐 token 全等"既苛刻又可达

你和 HF 都是 bf16 存储 + float 累加（torch CPU 的 opmath 也是 float），每个中间值差 ~1 ulp(bf16)。argmax 只看 top-1，而 top-2 gap 通常远大于数值噪声 → 128 步全等是**设计出来的可达目标**。

但确实存在某步 top-2 gap 极小、argmax 被噪声翻转的可能——翻转后序列全分叉。**先查 gap 再怀疑代码**：gap 大还分叉 = bug；gap 极小 = 可能是噪声（见"精度深入"）。

---

## 对拍工具

- 你侧：`tensor.debug()`（python）/ `tensorDebug`（C API）——打印 shape、strides、dtype、全量数据。
- HF 侧：`register_forward_hook` 抓每层输入输出，或 `output_hidden_states=True` 拿每层 hidden states。
- 对齐单位：HF 的 `layers[i]` 输出 == 你的第 i 层 `add` 完之后的 hidden。

---

## 排查顺序（从前往后，找第一个分叉）

```
0. embedding 输出（第 0 层输入）    —— ids 对吗？in_embed 加载对吗？
1. layer0 attn_norm 后             —— eps 是 1e-6 吗？
2. q/k（rope 前）                  —— bias 加了吗？view 形状对吗？
3. q/k（rope 后）                  —— pos_ids、θ、half-split
4. cache 写入后读出的 k            —— 写对位置了吗？
5. attn 输出                       —— GQA 映射、scale、causal
6. o_proj + 残差后                 —— o 无 bias！残差加的是 hidden 不是 normed
7. mlp 后（layer0 末）             —— gate/up 没反？down 无 bias
8. layer27 末 → final norm → logits —— 切最后一行了吗？
9. argmax                          —— logits view 成 1-D 了吗？
```

每层判定标准：bf16 下**不比逐位相等**——看最大相对差，1e-2 内算同（噪声量级），差出数量级才是 bug。第一个"差出数量级"的位置就是案发现场。

---

## 高频 bug 清单（按踩坑概率排序）

1. eps 用 1e-5（config 是 1e-6）
2. qkv bias 漏加 / o_proj 加 bias
3. rope 的 pos 从 0 开始（decode 时）/ 对 cache 重复 rope
4. GQA 映射写成 `h % nkvh`（正确：`h / (nh/nkvh)`）
5. gate/up 搞反（gate 过 sigmoid）
6. 残差漏加或加错对象
7. cache 读全量 maxseq
8. view 维序错（[s,12,128] vs [s,128,12]）
9. 返回漏 prompt / eos 处理错
10. 某个中间张量建成 f32

---

## 精度深入（可选）

若怀疑是噪声翻转而非 bug：把同一位置的 top-2 logits 打出来看 gap；gap ~1e-3 级（bf16 ulp）则两边都"对"、只是走了不同分支。验证法：换 f32 权重模型重跑对比。**注意分叉是 sticky 的**——一旦翻转不会自愈，所以"后面大部分一样"不能证明前面对。

---

## CI 验证

push 代码触发 Actions：Assignment-3 step 跑 `test_infer.py --test`（CI 自己从 HF 下载模型）。本地过了 CI 也可能挂：ubuntu + windows 双平台——未初始化变量、对齐假设等 UB 会在别的平台现形。

---

## 阅读路线（约 15 分钟）

1. `src/tensor/tensor.cpp` 的 debug/print_data——你的显微镜。
2. `test/test_infer.py` 全文——裁判规则。
3. site-packages 里 transformers 的 modeling_qwen2.py——金标准逐行对照。

**思考问题：**

- 每层算子单测都过、layer0 输出却对不上——最可能是哪类问题？（接线问题：顺序/对象/残差，而非 kernel）
- 第 57 步分叉但 top-2 gap 是 1e-4——bug 还是噪声？怎么验证？
- 为什么从 logits 倒推是低效的？

---

## 自测清单

- [ ] 默写排查顺序 10 步。
- [ ] 说出 HF hook 抓的 layers[i] 输出对应你侧哪个张量。
- [ ] 解释 argmax 翻转的两个来源及区分方法。

---

## 实现提示

- 模型里加 debug 开关：成员 `debug_layer_ = -1`（关），forward 每层末 `if (i == debug_layer_) hidden->debug()`；
- 写个独立对拍小脚本（不动 test）：HF hook 抓 layers[0] 输出存下来，和你 debug 打印的前几个数肉眼对；
- 首 token 对之前别开 decode；一次只改一处再测。

---

## 反面教材

- 凭直觉改代码碰运气。
- 从 logits 倒推。
- 对拍时要求 bf16 逐位相等（该比量级/相对差）。
- 一次改多处——修好了也不知道是哪处修的。

---

## 验证步骤

终验：`python test/test_infer.py --model /root/Model --test` → `Test passed!` → commit & push → CI Assignment-3 变绿。
