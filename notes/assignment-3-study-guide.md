# 作业 #3 学习指南：Qwen2 大模型推理

## 任务目标

`python test/test_infer.py --model /root/Model --test` 通过：你的 LLAISYS 推理结果与 HuggingFace PyTorch 参考**逐 token 全等**（test_infer.py:148 的 `assert llaisys_tokens == tokens`）。

`--test` 强制 `top_k=1, top_p=1.0, temperature=1.0`（test_infer.py:96-97），即**贪心采样 = argmax**，你作业#2 的 argmax 算子直接够用。

三条硬约束（README_ZN.md:293-315）：
- 推理逻辑必须在 C/C++ 后端，Python 侧**禁止**用 torch/numpy 算 forward；
- 必须实现 KV-Cache，否则慢到不可用；
- C API 原型已在 `include/llaisys/models/qwen2.h` 给出：create / destroy / weights / infer 四个函数。

---

## 全局图景：一次 generate 发生了什么

`llaisysQwen2ModelInfer(model, token_ids, ntoken)` 的签名（qwen2.h:40）透露了整体设计：

- 输入是**一组** token（prefill 时是整个 prompt，decode 时是上一步产出的 1 个 token）；
- 返回值是**单个** int64 —— 一次 infer 只前向一次、采样一个 token；
- 因此**位置状态（past_len）必须存在 C++ 模型对象内部**，每次 infer 后 `past_len += ntoken`。

Python 的 generate 循环骨架（自己补全细节）：

```
next_id = infer(prompt_tokens)        # prefill：整段 prompt 进模型，产出第 1 个新 token
outputs = [next_id]
while next_id != end_token 且未达 max_new_tokens:
    next_id = infer([next_id])        # decode：每次只进 1 个 token
    outputs.append(next_id)
```

---

## 模型档案（/root/Model/config.json 实测值）

对应 `LlaisysQwen2Meta`（qwen2.h:7-12）的填法：

| meta 字段 | config.json 字段 | 值 |
|---|---|---|
| dtype | torch_dtype | BF16 |
| nlayer | num_hidden_layers | 28 |
| hs | hidden_size | 1536 |
| nh | num_attention_heads | 12 |
| nkvh | num_key_value_heads | 2 |
| dh | hs / nh | 128 |
| di | intermediate_size | 8960 |
| voc | vocab_size | 151936 |
| epsilon | rms_norm_eps | 1e-6 |
| theta | rope_theta | 10000 |
| end_token | eos_token_id | 151643 |
| maxseq | （自己定，见思考问题） | — |

GQA 分组 = nh/nkvh = **6**（12 个 q 头共享 2 个 kv 头，正是你 2.6 里 `group = nh/nkvh` 的用武之地）。

---

## 权重映射表（safetensors 名字 → LlaisysQwen2Weights）

`LlaisysQwen2Weights`（qwen2.h:14-30）里单层字段是**指针数组**，`attn_q_w[i]` 是第 i 层。共 339 个张量 = 2 个嵌入 + 1 个 final norm + 28 层 × 12 个：

| safetensors 名字（第 i 层） | 形状 | weights 字段 |
|---|---|---|
| model.embed_tokens.weight | [151936, 1536] | in_embed |
| lm_head.weight | [151936, 1536] | out_embed |
| model.norm.weight | [1536] | out_norm_w |
| layers.i.input_layernorm.weight | [1536] | attn_norm_w[i] |
| layers.i.self_attn.q_proj.weight | [1536, 1536] | attn_q_w[i] |
| layers.i.self_attn.q_proj.bias | [1536] | attn_q_b[i] |
| layers.i.self_attn.k_proj.weight | [256, 1536] | attn_k_w[i] |
| layers.i.self_attn.k_proj.bias | [256] | attn_k_b[i] |
| layers.i.self_attn.v_proj.weight | [256, 1536] | attn_v_w[i] |
| layers.i.self_attn.v_proj.bias | [256] | attn_v_b[i] |
| layers.i.self_attn.o_proj.weight | [1536, 1536] | attn_o_w[i]（**无 bias**） |
| layers.i.post_attention_layernorm.weight | [1536] | mlp_norm_w[i] |
| layers.i.mlp.gate_proj.weight | [8960, 1536] | mlp_gate_w[i] |
| layers.i.mlp.up_proj.weight | [8960, 1536] | mlp_up_w[i] |
| layers.i.mlp.down_proj.weight | [1536, 8960] | mlp_down_w[i] |

注意：**qkv 有 bias，o_proj 没有**（Qwen2 的特点，你 2.3 的可选 bias 就是为它准备的）；256 = nkvh×dh；`tie_word_embeddings=false`，所以 lm_head 独立存在，不能用 embed_tokens 顶替。

加载链路：python 空壳（models/qwen2.py:16-20）已经打开了 safetensors → 拿到 numpy 数组 → 按上表找到对应的 `llaisysTensor_t`（通过 `llaisysQwen2ModelWeights` 拿结构体指针）→ `tensorLoad` 把数据拷进去。

---

## 单层 forward 数据流（全是你作业#2 的算子）

设 seqlen=s，past=past_len。输入 hidden [s, 1536]（首层来自 embedding 算子查表 in_embed）：

```
1. normed = rms_norm(hidden, attn_norm_w[i], eps)            # [s, 1536]
2. q = linear(normed, q_w, q_b)  → view [s, 12, 128]         # 有 bias
   k = linear(normed, k_w, k_b)  → view [s, 2, 128]
   v = linear(normed, v_w, v_b)  → view [s, 2, 128]
3. rope(q, pos_ids, theta); rope(k, pos_ids, theta)          # pos_ids = [past, ..., past+s-1]
4. kcache[i][past : past+s] = k    # 写 KV cache（先 rope 再写！）
   vcache[i][past : past+s] = v
5. attn = self_attention(q, kcache[i][0:past+s], vcache[i][0:past+s], scale=1/√128)
                                        # → [s, 12, 128] → view [s, 1536]
6. hidden = add(hidden, linear(attn, o_w))                   # 残差；o 无 bias
7. normed = rms_norm(hidden, mlp_norm_w[i], eps)
8. hidden = add(hidden, linear(swiglu(linear(normed, gate_w), linear(normed, up_w)), down_w))
```

28 层走完后：

```
9. hidden = rms_norm(hidden, out_norm_w, eps)
10. logits = linear(hidden[最后一行], out_embed)             # [1, 151936]
11. next = argmax(logits)                                    # 返回 int64
```

---

## 关键设计决策（先想通再写代码）

1. **采样只需要最后一个 token 的 logits**：第 10 步前把 hidden 切出最后一行 [1, 1536] 再做 norm + lm_head。lm_head 是 [151936, 1536] 的大家伙，对整段 s 个位置算是纯浪费。
2. **KV cache 形状 [maxseq, nkvh, dh]**，按层预分配。写 cache：`slice(0, past, past+s)` 沿 dim 0 切片**结果仍连续**（想想为什么），可以直接 memcpy 或实现 rearrange 后调用。读 cache：`slice(0, 0, past+s)` 同样连续，self_attention 直接吃。
3. **k 先 rope 再写 cache**：cache 里存的是旋转后的 k，旧位置永远不能重复旋转。
4. **causal mask 不用你管**：self_attention 内部用 `j ≤ kvlen − qlen + i` 处理（回看 README_ZN.md:243-265 和你 2.6 的实现），模型层只要把 qlen、kvlen 传对。
5. **rearrange 还是空壳**（src/ops/rearrange/op.cpp:5）。最小路径用 memcpy（写 cache 的目标是连续内存）；顺手实现 rearrange 是加分项（它就是 strided copy）。
6. **dtype 全链路 BF16**：权重、中间激活都建 BF16 tensor。算子内部已做 float 累加，精度够和 HF bf16 参考对齐。
7. **pos_ids 是 Int64 张量**（rope 算子的 CHECK 会查），内容为 [past, past+1, ..., past+s-1]，用 tensor.load 从主机内存灌进去。

---

## 代码组织（README 硬性要求）

- C++ 模型实现：`src/models/qwen2.cpp`（新模块）——内部持有一组 tensor_t 权重、KV cache、past_len；forward 全部调 `llaisys::ops::*`。
- C API：`src/llaisys/models.cc`（llaisys target 已 glob `src/llaisys/*.cc`，仿照 ops.cc 的写法，opaque struct 包一层）。
- xmake.lua：给 `src/models/*.cpp` 加一个 static target（仿照 llaisys-ops 的写法），并挂进 `target("llaisys")` 的 add_deps。
- ctypes 包装：`python/llaisys/libllaisys/` 下新建文件仿照 ops.py 写 `load_qwen2(lib)`，在 `__init__.py` 里注册（注意 meta/weights 结构体要用 ctypes.Structure 镜像定义）。
- `python/llaisys/models/qwen2.py`：__init__ 读 config.json 填 meta、create 模型、遍历 safetensors 按映射表 tensorLoad；generate 写上面的循环。

---

## 阅读路线（约 30 分钟）

1. `test/test_infer.py:96-149` — 判定标准与 generate 的调用方式（top_k/top_p/temperature 参数怎么传进来）。
2. `include/llaisys/models/qwen2.h` — 四个函数 + 两个结构体，这是契约。
3. `python/llaisys/models/qwen2.py` — 空壳，注意 safetensors 遍历已经写好了。
4. `src/llaisys/ops.cc` — C API 如何包 C++（`x->tensor` 模式）。
5. `python/llaisys/libllaisys/ops.py` + `__init__.py:37-40` — ctypes 注册模式。
6. `python/llaisys/tensor.py` — Python 侧 Tensor 可用方法；C API 侧对应 tensorCreate/tensorLoad（libllaisys/tensor.py）。
7. `src/tensor/tensor.cpp` — load / slice / view 的实现，KV cache 三板斧。
8. `xmake.lua` 末尾的 `target("llaisys")` — 看依赖怎么挂。

**思考问题：**

- prefill 和 decode 时 ntoken 分别是多少？pos_ids 各怎么生成？
- 为什么 k 要先 rope 再写 cache，而不是用的时候再 rope？
- KV cache 沿 dim 0 的 `slice(0, past, past+s)` 为什么连续？slice 哪一维会破坏连续？
- maxseq 设成 config 里的 131072 要吃多少内存？自己算：28 层 × 2(k,v) × maxseq × 2 头 × 128 × 2 字节。设多少合理？
- lm_head 一步 decode 的 FLOPs 约 2×1536×151936 ≈ 0.47 GFLOP，单层 attn+mlp 约多少？感受"只为最后一个 token 算 logits"省了多少。
- decode 时 qlen=1，self_attention 的 causal 条件退化成什么？
- generate 什么时候停止？返回的 outputs 包含 prompt 吗？（对一下 test_infer.py:148 怎么比）

---

## 自测清单（能回答才算读懂）

- [ ] 默写 12 个权重名 → 字段 → 形状的对应。
- [ ] 画出第 0 层完整数据流并标注每个中间张量的 shape。
- [ ] 解释 infer 为什么输入是数组、返回却是单个 token。
- [ ] 解释为什么 `--test` 下 top_k=1 就等价于 argmax 算子。
- [ ] 解释 bf16 下为什么你的结果能和 HF 逐 token 相等（提示：argmax 对微小扰动的鲁棒性 + 两边都 bf16 存储/float 累加）。

---

## 实现顺序建议（四个里程碑）

- **M1 骨架**：meta/weights/create/destroy 打通，python 加载全部 339 个权重，逐层打印 shape 和 safetensors 对齐。
- **M2 prefill 单步**：整段 prompt 前向 + argmax，对比 HF 输出的第一个新 token（HF tokens 列表里 prompt 之后的第一个）。
- **M3 decode 循环**：接通 KV cache，逐 token 生成 128 步，肉眼读输出是否通顺。
- **M4 对拍**：`--test` 全等。不一致时用 `tensorDebug`（tensor.py 的 debug）从第 0 层的 normed/q/k/attn 逐一和 HF hook 抓的中间值对——**从前往后找第一个分叉点**，不要从 logits 倒推。

---

## 反面教材（不要这么写）

- 在 Python 里用 numpy/torch 算任何一层 forward —— README 明令禁止。
- 每次 decode 都重算整段序列 —— 没有 KV cache，README 点名不行。
- 对 cache 里的旧 k 重复 rope。
- q/k/v 忘加 bias，或给 o_proj 加 bias。
- 权重 load 成 f32 tensor —— CHECK_SAME_DTYPE 会挂，必须和 meta.dtype 一致（BF16）。
- final norm + lm_head 对整段 seqlen 做再取最后一行 —— 不算错但 decode 慢一个数量级。
- 忘了 infer 内部维护 past_len，或 generate 忘了把新 token 喂回下一步。
- sampling 过度设计：C API 只返回 int64，`--test` 只需要 argmax；top_p/temperature 全支持不是本作业的判定点。

---

## 验证步骤

```bash
mamba activate llaisys   # 或用全路径 /root/miniforge3/envs/llaisys/bin/{xmake,python}
XMAKE_ROOT=y xmake && XMAKE_ROOT=y xmake install
python test/test_infer.py --model /root/Model --test
```

**预期现象**：先打印 HF 的 Answer（tokens + 文本），再打印 Your Result，两者 token 列表完全一致，最后 `Test passed!`。之后 commit & push，看 GitHub Actions 的 Assignment-3 步骤变绿。

---

## 学习建议

- 动手前默画一遍整体数据流（含 shape），把 2.1–2.7 每个算子标在它出现的位置——体会"算子是积木，模型是搭法"。
- 做完后回看 self_attention 的 causal 条件 `j ≤ kvlen − qlen + i` 和 KV cache 的关系：prefill（qlen=kvlen）时它退化成普通下三角，decode（qlen=1）时全部可见——一个式子覆盖两种场景，值得品味。
- 这个作业调通的那一刻，你就拥有了一个亲手从张量开始搭出来的、能跑的 1.5B 推理引擎。
