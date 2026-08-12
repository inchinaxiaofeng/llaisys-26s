# 任务 3.2 学习指南：把 3.1 GB 权重灌进模型

## 任务目标

`Qwen2("/root/Model")` 结束后 339 个张量全部就位，抽查数值与 safetensors 里的原始数据**逐字节一致**。

---

## 核心坑：numpy 读不了 bf16（已实测）

空壳代码用 `safetensors.safe_open(..., framework="numpy")`，但 numpy 没有 bf16 dtype，`get_tensor` 直接炸：

```
TypeError: data type 'bfloat16' not understood
```

解法：换 `framework="pt"` 拿 torch 张量，再 `view(torch.uint16).numpy()` 得到**共享内存**的 uint16 视图——零拷贝取出原始字节：

```python
t = f.get_tensor(name)                 # torch.bfloat16
a = t.view(torch.uint16).numpy()       # uint16 视图，同一块内存
LIB_LLAISYS.tensorLoad(handle, ctypes.c_void_p(a.ctypes.data))
```

这违反"禁止 python 框架"吗？不违反——约束针对**推理逻辑**（forward 计算），权重加载只是字节搬运，test_infer.py 自己也 import torch。（备选硬核路线：自己解析 safetensors 文件头的 JSON 拿字节偏移，`np.frombuffer` 读原始字节，完全不依赖 torch——不必，但值得知道。）

---

## tensorLoad 链路（数据是怎么进模型的）

`Tensor.load(c_void_p)`（python/llaisys/tensor.py:75-76）→ `tensorLoad`（src/llaisys/tensor.cc:68-72）→ `Tensor::load`（src/tensor/tensor.cpp:293-302）→ `memcpy_sync(H2D)`，拷贝 `numel × elementSize` 字节。

两个推论：

1. **dtype 必须在 tensorCreate 时就定对**（BF16）——load 只搬字节不解释，bf16 字节喂给 f32 张量就是垃圾数据；
2. CPU 模型里 H2D 实际是 host→host 拷进 tensor 自己的 storage，load 返回后 numpy/torch 侧的数组就可以释放。

---

## 映射表代码化

顶层 3 个 + 逐层 12 个（完整表见总指南"权重映射表"节）。建议写一个 route 函数：

```
name 以 "model.layers." 开头 → 解析出层号 i 和尾缀（如 self_attn.q_proj.weight）
    尾缀查表 → (字段名, i) → getattr(w, 字段名)[i]
否则 → 顶层查表 → w.in_embed / w.out_embed / w.out_norm_w
```

`getattr(w, "attn_q_w")[i]`——ctypes 的 POINTER 字段直接支持下标。

---

## 加载循环

- 遍历 `sorted(model_path.glob("*.safetensors"))`（空壳已写好），**每个文件内**遍历 `f.keys()`；
- 每 load 一个打印进度（339 个张量，静默几分钟会以为死机）；
- 加载完打印总数对账：2 嵌入 + 1 norm + 28×12 = **339**。

---

## 内存

权重总量 ≈ 1.5B 参数 × 2 B ≈ 3.1 GB；单张量最大是 embed/lm_head（[151936,1536]×2B ≈ 444 MB）。逐张量拷贝，峰值 = 模型本体 + 最大张量的双份，4 GB 出头，本机无压力。

---

## 阅读路线（约 15 分钟）

1. `python/llaisys/models/qwen2.py:14-20` — 空壳的遍历框架。
2. `include/llaisys/tensor.h:48-50`、`src/llaisys/tensor.cc:68-72`、`src/tensor/tensor.cpp:293-302` — load 链路三段。
3. `python/llaisys/libllaisys/tensor.py` — tensorLoad 的 argtypes。
4. `test/test_utils.py:5-34` — random_tensor 里另一种拿指针的方式（data_ptr）。

**思考问题：**

- 为什么用 `view(torch.uint16)` 而不是 `.to(torch.float32)`？两条路的数据和内存各发生了什么？
- tensorLoad 的 H2D 在纯 CPU 模型里实际是什么操作？
- 若换一个 f32 权重的模型，这套加载代码要改哪几处？
- 验算 339 = 3 + 28×12：每层的 12 个是哪些？

---

## 自测清单

- [ ] 默写映射表（safetensors 名 → 字段）。
- [ ] 说出 bf16 字节流经的每一步：文件 → torch tensor → uint16 视图 → numpy → void* → memcpy。
- [ ] 解释为什么"先转 f32 再 load"既错精度又翻倍内存。

---

## 实现提示

- 先只加载顶层 3 个 + 第 0 层 12 个，打印 shape/dtype 对账，再放开全量循环。
- 抽查验证：load 后用 tensorDebug 打印 `w.attn_q_b[0]`，和 torch 直接打印 `f.get_tensor("model.layers.0.self_attn.q_proj.bias")` 的前几个数对比——bf16 值应完全一致（同一份字节）。
- route 查不到的名字直接 raise——宁可崩在加载期，不要静默漏权重。

---

## 反面教材

- framework="numpy" 硬读 bf16 → TypeError。
- `.to(torch.float32)` 再存 f32 张量 → dtype 与 meta 不符，算子 CHECK_SAME_DTYPE 挂；内存翻倍。
- 靠文件名/字典序推断层号（应从张量名 parse）；多 shard 时顶层张量重复 load（本模型单文件，但写法要鲁棒）。
- 查不到映射就 pass——静默漏权重，forward 时才炸。

---

## 验证步骤

`Qwen2("/root/Model")` 跑完打印 `loaded 339/339`；抽查 3 个张量（一个 bias、一个大矩阵的角、一个 norm 权重）数值与 torch 一致。
