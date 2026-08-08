# 任务 1.3 学习指南：实现 `Tensor::view`

## 任务目标

实现 `src/tensor/tensor.cpp`（207-210 行）中的：

```cpp
tensor_t view(const std::vector<size_t> &shape) const;
```

语义：**零拷贝 reshape**——共享同一块 Storage，只换一套元数据（shape + strides），把同一串字节换一种读法。

README 的两个硬性要求（缺一都不算完成）：

1. 新 shape 的元素总数必须等于原 numel；
2. 新视图与原始布局**不兼容时必须抛错**。README 的例子：shape (2,3,5)、strides (30,10,1) 的张量，能不能不搬数据就变成 (2,15)？

### 补充思考：

view和reshape的区别在于对contiguity的要求，以及是否会触发内存拷贝。
view是更严格的Zero Copy“View”，不连续就报错；
reshape则是更智能的“形状改变”，能Zero Copy就ZeroCopy，不能就自动拷贝内存。

---

## 核心洞察：为什么 view 要求连续？

把"合并维度"想清楚：(3,5) 合并成 15，意味着新下标 m∈[0,15) 要满足 m = j×5 + k。要零拷贝，物理地址必须能写成 `base + m × s`（s 是某个单一 stride）。

展开看：原地址 = j×stride\[0\] + k×stride\[1\]。当且仅当 **stride\[0\] == 5 × stride\[1\]**（即 stride\[i\] == stride\[i+1\] × shape\[i+1\]，被合并的维在原布局里紧挨着），地址才能写成 (5j+k)×stride\[1\] = m×stride\[1\]。

——眼熟吗？**这正是 1.2 isContiguous 的逐维判据**。连续张量对任意合并/拆分都满足这个条件，所以本项目的简化规则就是：view 要求原张量连续，新 strides 按新 shape 重新紧凑计算。

**手算 README 的例子**（务必动笔）：shape (2,3,5)、strides (30,10,1)，第 0 行的 15 个逻辑元素的物理位置是 {0..4, 10..14, 20..24}——不是连续段，任何单一 stride 都描述不了 → 必须抛错。

---

## 阅读路线（约 20 分钟）

1. `src/tensor/tensor.cpp:14-43` — `create`：新 strides 的紧凑算法你已经抄过一遍了。
2. `src/tensor/tensor.cpp:173-200` — 你刚写的 `isContiguous`：本任务的第一道校验。
3. `src/tensor/tensor.cpp:207-210` — view 的 stub：**盯着它的 return 看，想想丢了多少东西**。
4. 调用链（已熟悉，快速过）：`tensor.py:81-85` → `tensor.cc:74-80` → `Tensor::view`。
5. `test/test_tensor.py:21-29` — view(6,10) 后的断言：shape、strides、check_equal。

**思考问题：**

- 新 Tensor 的 `_offset` 该传什么？stub 的 `new Tensor(_meta, _storage)` 犯了几个错？（提示：对一个做过 dim-0 切片的张量 view 一下会发生什么）
我理解是，dim-0切片在view之后，继承切片的_offset就好。
✅ 对——`new Tensor(meta, _storage, _offset)`，offset 原样继承。但问题还有一半没答：stub 的**第二个**错是 `_meta` 整个没换（shape/strides 还是旧的），等于什么都没做。


- 校验 numel：新 shape 的元素总数怎么算？（`numel()` 里有现成的 `std::accumulate` 写法）项目里现成的校验宏是哪个？（翻 `src/utils/check.hpp`，找 `CHECK_ARGUMENT`）

对新元素用`numel`?

✅ 订正（上一条 numel 问题）：`numel()` 是成员函数，算的是**旧**张量（this）的元素数。新 shape 还只是个 `std::vector`，没有张量可调用——要自己照抄 `numel()` 里的写法算乘积：`std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())`，再和 `this->numel()` 比较。校验宏是 `CHECK_ARGUMENT(condition, message)`（check.hpp:21-28）。

- PyTorch 的 `view(-1)`（自动推断维度）在这个项目里能实现吗？（看 `tensor.cc:74-80` 的形参类型 `size_t *`）

view(-1)是将一个任意维度的Tensor 展平成 1D Tensor的操作。-1表达“自动推导维度”。
但是能不能设计出来还真跟怎么做不太有关。

size_t是 unsigned int，无法表达负数。但是有一个方式：在C/C++的补码机制下，给size_t类型赋 -1，会变成SIZE_MAX，然后在Tensor::view()的代码中去捕捉特定值就好。
但是这种设计更多是“Hack”而不是设计，特别是这个依赖特定编译器，属于UB。因此，好的代码不应该这么设计。
✅ 订正一处：`size_t x = -1;` 在 C++ 里其实是**定义良好**的模 2^N 转换（保证得到 SIZE_MAX），不是 UB。但它依然是糟糕的 API 设计（魔法哨兵值、ABI 语义不清）。真正正确的做法：若要支持 -1 推断，在 **Python 包装层**先算好推断维度，把显式 shape 传给 C API——底层接口保持简单。

- 在"要求原张量连续"的前提下，view 的结果一定连续吗？为什么？
我认为一定连续。原Tensor就是一个“毫无空隙、行优先连续排列的内存区间（逻辑索引和物理Offset严格线性对齐），而view的唯一工作就是根据new_shape计算出一套标准的连续strides。
这个是从定义的角度出发的。
✅ 正确：输入连续 + strides 按新 shape 紧凑重算，结果连续是构造出来的。

- view 和 reshape（挑战任务）的关系？（一句话：reshape = 能 view 就 view，否则先 contiguous 再 view）
view和reshape的区别在于对contiguity的要求，以及是否会触发内存拷贝。
view是更严格的Zero Copy“View”，不连续就报错；
reshape则是更智能的“形状改变”，能Zero Copy就ZeroCopy，不能就自动拷贝内存。

- view 是 const 方法，但新 Tensor 与 this 共享 Storage——通过新 Tensor 写数据，this 会看到吗？（联系 1.1 笔记 Tensor/Storage 分离的第 4 点）

应该是会看到的，因为Tensor是对数据的描述，而new tensor和current tensor描述的是一个storage。storage发生改变，this当然会看到。
一个更进一步的思考是，reshape会看到吗？我理解这个答案就是，如果是zero copy的reshape，就可以看到，反之不行。
✅ 正确，reshape 的推论也对：zero-copy 分支共享 Storage 所以可见；拷贝分支落到新 Storage 上，与原张量无关。

---

## 自测清单（能回答才算读懂）

- [ ] 一句话：view 改了什么、没改什么？
view 改变了Meta Data，但是没有改变数据在内存中的位置(Zero Copy)，没有改变数据的contiguous，没有改变数据本身。

- [ ] 用 strides 语言写出"两个维度可以零拷贝合并"的条件。

Strides\[i\] = ∏^{N-1}_{k+1}Shape\[k\]

✅ 订正（上一条合并条件问题）：方向对但答偏了——题目问的是**相邻两维**可合并的**局部**条件：`stride[i] == stride[i+1] × shape[i+1]`。你写的是整体连续时 stride 的展开式（更强的全局条件，也是本项目实际采用的前提）。另外求和下标笔误：k 应从 i+1 开始。

- [ ] (3,4,5) view 成 (6,10)，新 strides 是多少？view 成 (2,2,3,5) 呢？
原本的shape(3,4,5)的strides是(20,5,1), shape(6，10)的strides是(10,1)
✅ (6,10) 对。漏了 (2,2,3,5)：**(30,15,5,1)**。

- [ ] 两道校验（连续性、numel）的先后顺序有影响吗？
没有影响吧？
✅ 功能上确实无影响——两个校验互相独立，只影响报错信息的先后。习惯上先验入参（numel）再验对象状态（连续性）；本项目测试对顺序无要求。

- [ ] 为什么不能直接修改 this 的 `_meta` 再返回 this？（至少说出两个原因）

1. 不带'_'的函数都是非原地操作，不应该改变输入x。
2. 这么修改不能支持“同一块物理内存的多个view共存”，深度学习网络中常常需要对同一个Tensor以不同维度视角同时计算。
3. C++ const 成员函数语法约束，头文件签名是`tensor_t Tensor::view(const std::vector<size_t> &shape) const;
`，在const函数内部，this指针类型是const Tensor*，编译器会禁止在函数内部修改_meta。
✅ 三点都成立。再补一个最致命的：就算用 const_cast 绕过语法限制改了 `_meta`，`return this` 也过不了类型关——裸指针构造 shared_ptr 会制造第二个 owner → double free（上次 contiguous 报的错）。而且测试后面还要对**原始** `llaisys_tensor` 继续做 permute/slice 断言，改了它后面全崩。

---

## 实现提示

骨架（细节自己填，正文约 10 行）：

1. `CHECK_ARGUMENT(isContiguous(), ...)` —— 非连续不能零拷贝 view；
2. 算新 shape 的元素总数，`CHECK_ARGUMENT` 它与 `numel()` 相等；
3. 照抄 `create` 里那段倒序循环，为新 shape 算紧凑 strides；
4. 组装 `TensorMeta{dtype, shape, strides}`，`return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));`

**反面教材（不要这么写）：**

- `new Tensor(_meta, _storage)` —— meta 没换、offset 丢了，双错；
- 只换 shape 不重算 strides —— 新旧 shape 不同，strides 几乎必然对不上；
- 任何 memcpy / 新 Storage —— view 必须零拷贝；
- 留着 stub 的 return 当占位忘了删。

---

## 验证步骤

```bash
xmake --root && xmake install --root
cd test && python test_tensor.py
```

**预期现象**：`===Test load===`、`===Test view===`（debug 打印 6×10 矩阵）全过，然后在 `===Test permute===` 抛 `TO_BE_IMPLEMENTED`——这就是 1.3 完成的标志，permute 是 1.4 的内容。

---

## 学习建议

- 想通"被合并的维必须紧挨"这个条件后回看 1.2：isContiguous 的地基在这里第一次承重。
- 记住等式 **reshape = view if possible else contiguous().view()**——挑战任务到那时只剩工程细节。
