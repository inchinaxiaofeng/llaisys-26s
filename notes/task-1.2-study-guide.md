# 任务 1.2 学习指南：实现 `Tensor::isContiguous`

## 任务目标

实现 `src/tensor/tensor.cpp`（172-175 行）中的：

```cpp
bool isContiguous() const;
```

语义：**只检查 shape 和 strides 这两个元数据**，判断张量的逻辑元素在物理内存中是否"连续摆放"。

- 纯元数据判断：不分配内存、不拷贝数据、不碰设备，因此也**不需要** setDevice / runtime。
- 这个函数是整个作业 #1 的地基：1.3 `view` 的合法性检查、挑战任务 `contiguous()` / `reshape()`、作业 #2 所有算子的扁平指针算术，全都建立在"连续"的定义上。

---

## 背景：到底什么叫"连续"？

按**行主序（row-major）**遍历逻辑坐标 `(i_0, i_1, ..., i_{n-1})` 时，如果对应的物理地址**严格递增、无空洞、无重叠**，这个张量就是连续的。

回忆 1.1 里 `Tensor::create`（tensor.cpp:27-30）生成 strides 的倒序累乘：

```
strides[ndim-1] = 1
strides[k] = strides[k+1] * shape[k+1]     （从右往左）
```

**这就是"连续布局"的定义本身**。`isContiguous` 要做的就是反过来问：当前 `_meta.strides` 是否等于"假如这个 shape 是紧凑布局时应该有 strides"？

手动跑两个例子（拿出纸笔，各算一遍）：

| 张量 | shape | strides | 连续？ |
|---|---|---|---|
| 原始 | (3,4,5) | (20,5,1) | ? |
| permute(2,0,1) 后 | (5,3,4) | (1,20,5) | ? |
| slice(2,1,4) 后 | (3,4,3) | (20,5,1) | ? |

注意第三行：strides 和原始张量**完全相同**，但它不连续——每行只取 3 个元素，行与行之间有空洞。这说明"连续"判断必须结合 shape 和 strides 一起推，光看 strides"是不是递减"会漏掉这种情况。

---

## 阅读路线（约 20 分钟，本任务代码量极小）

1. `src/tensor/tensor.hpp:9-13` — `TensorMeta`：这个函数能用的全部信息就是 `dtype / shape / strides`。
2. `src/tensor/tensor.cpp:27-30` — `create` 里 strides 的生成方式（连续的定义）。
3. `src/tensor/tensor.cpp:172-175` — 当前 stub。
4. `test/test_tensor.py` — 找到所有 `is_contiguous` 断言（18、28、38、48 行），**仔细看断言的对象是谁**。
5. 调用链（1.1 已学过，快速过）：`python/llaisys/tensor.py:78-79` → `src/llaisys/tensor.cc:63-66` → `Tensor::isContiguous`。

**思考问题：**

- `_offset` 影响连续性吗？（提示：`tensor[1:, :]` 这种 dim-0 切片，offset 增加了、strides 没变，它连续吗？连续性的定义里出现过"起点"吗？）

- size-1 的维度：shape `(2,1,3)`、strides `(3,1,1)` 连续吗？如果 strides 是 `(3,999,1)` 呢？
  （提示：size-1 的维度上只能取索引 0，它的 stride 永远不会被用到。PyTorch 认为两者都连续。测试里没有 size-1 维的用例，严格版算法能过测试——但想想哪个行为更"正确"，要不要顺手处理。）

- 0 维张量（shape 为空）和空张量（某维为 0）应该返回什么？

- strides 的类型是 `ptrdiff_t`（有符号）。负 stride 意味着什么？这个项目支持吗？（看 `test/test_utils.py:132-133`。）

- 如果 `isContiguous()` 直接 `return true;`，`test_tensor.py` 能过吗？为什么仍然不允许？
  （提示：看 18/28/38/48 行断言的永远是哪个张量；再想想作业 #2 的算子为什么敢对 `data()` 做扁平指针算术。）

---

## 自测清单（能回答才算读懂）

- [ ] 用一句话写出连续的判据（涉及 shape、strides，以及一个从 1 开始累乘的期望值）。
- [ ] 为什么判断顺序要**从最后一维往前**推，而不是反过来？
- [ ] 为什么 `_offset` 和连续性无关？
- [ ] permute 之后为什么一定不连续？（有例外吗？想想 size-1 维或对调后形状不变的特殊情况——不用实现，理解即可。）
- [ ] slice 之后可能连续吗？（提示：沿 dim 0 切 vs 沿最后一维切，有区别。）

---

## 实现提示

骨架（细节自己填，正文约 5-8 行）：

1. 从最后一维开始，维护一个"期望 stride"，初始为 1；
2. 逐维向前：比较当前维的 stride 与期望值，不等则返回 false，否则期望值乘上当前维的大小；
3. 全部通过则返回 true。

**反面教材（不要这么写）：**

- `return true;` —— 能过 `test_tensor.py`（因为断言的永远是原始张量），但 1.6 之后和作业 #2 会全部崩掉，而且这违背任务语义；
- 只检查 strides 是否单调递减 —— 会被 slice 的反例骗过（shape (3,4,3) strides (20,5,1) 递减但不连续）；
- 用 `_offset == 0` 判断 —— offset 与连续性无关（想想 dim-0 切片）；
- 调 `setDevice` / `runtime()` —— 纯元数据判断，碰上下文是多余的。

---

## 验证步骤

```bash
xmake --root && xmake install --root
cd test && python test_tensor.py
```

（editable 安装已配好，改 C++ 后只需这两步，不用再碰 pip。）

**预期现象**：`===Test load===` 完整通过（包括第 18 行的 `is_contiguous` 断言），然后在 `===Test view===` 抛 `TO_BE_IMPLEMENTED` 异常——**这就是 1.2 完成的标志**，view 是任务 1.3 的内容。

非连续分支暂时没法用 Python 测（permute/slice 还没实现）。验证方式：

1. 用纸笔手推上面表格里的三个例子，和代码逻辑对照；
2. 记住这个欠账：**1.6 跑通完整测试后，回来补验 permute/slice 视图上的 `is_contiguous()`**（可以写个小脚本和 `torch.Tensor.is_contiguous()` 对拍）。

---

## 学习建议

- 这个任务分值全在"定义"上：连续不是属性，是 **shape 与 strides 之间的一个关系**。
- 顺手想清楚：`view`（1.3）为什么要求张量连续？README 任务 1.3 里那个 shape (2,3,5)、strides (30,10,1) 的例子，为什么不能零拷贝地 view 成 (2,15)？想通这个，1.3 就完成了一半。
