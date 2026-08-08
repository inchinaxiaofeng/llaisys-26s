# 任务 1.4 学习指南：实现 `Tensor::permute`

## 任务目标

实现 `src/tensor/tensor.cpp`（204-207 行）中的：

```cpp
tensor_t permute(const std::vector<size_t> &order) const;
```

语义：**重排维度的顺序**，零拷贝。转置（transpose）就是 permute 的特例。测试用例：(3,4,5) 做 `permute(2, 0, 1)` → shape (5,3,4)、strides (1,20,5)。

这是三个 Meta Transform 里最简单的一个：**shape 和 strides 按同一个排列重排，offset 不动，Storage 不动**。

---

## 核心：搞清 `order` 的约定方向

`order` 的语义是：**新张量的第 i 维 = 原张量的第 order\[i\] 维**，即：

```
new_shape[i]   = shape[order[i]]
new_strides[i] = strides[order[i]]
```

用测试用例验证这个方向（务必动笔）：

- 原 shape (3,4,5)，order = (2,0,1)
- new_shape\[0\] = shape\[2\] = 5，new_shape\[1\] = shape\[0\] = 3，new_shape\[2\] = shape\[1\] = 4 → (5,3,4) ✓ 和 torch 一致
- 原 strides (20,5,1) → 同样按 order 重排 → (1,20,5) ✓

注意反方向 `new_shape[order[i]] = shape[i]` 是**逆排列**，对本例恰好也……算算看：(4,5,3)——和 torch 的 (5,3,4) 对不上，测试能抓住这个错误。但某些对称的 order（如 (1,0)）两种写法结果一样，所以别只靠 (1,0) 脑补验证。

---

## 阅读路线（约 15 分钟）

1. `test/test_tensor.py:31-39` — permute 节的断言：shape、strides、check_equal 全都有，这次的变换结果会被严格验证。
2. `python/llaisys/tensor.py:87-90` — Python 层已经帮你校验了什么？（只有长度）C++ 层还应该校验什么？
3. `src/llaisys/tensor.cc:82-87` — 注意它怎么拿 ndim 的。
4. `src/tensor/tensor.cpp:204-207` — stub 本体。
5. 顺便看一眼 `print_data`（tensor.cpp:106-121）：它是按 strides 递归寻址的，所以 permute 后的非连续张量也能打印出逻辑正确的内容——你的 1.2 在这里再次承重。

**思考问题：**

- permute 后 `_offset` 变吗？为什么？（提示：起点那个元素 (0,0,...,0) 在物理内存里动了吗？）
不需要变化，Slice之后的去permute，肯定不会有问题的
> ✅ 结论对。把理由说透：元素 (i,j,k) 的物理地址 = storage + offset + i·s₀ + j·s₁ + k·s₂，permute 只是重排"哪根轴配哪个 stride"，任何元素的物理位置都没动，起点 (0,…,0) 仍在 offset 处。所以新 Tensor 原样继承 `_offset`（slice 产生的非零 offset 也照传）——你说"slice 后再 permute 没问题"正是这个原因。
- `permute(1, 0)` 等价于什么操作？
转置 ✅
- permute 的结果连续吗？（联系 1.2 自测题）什么情况下仍然连续？
不一定连续。如果是一个1Dim Tensor，不知道算不算连续（因为就说不上这个操作了。同理还有空Tensor和标量。
> ✅ "不一定连续"对。补充"什么时候仍连续"：① 恒等排列（含 1-D 唯一的 permute(0)）当然连续；② 空 Tensor / 标量按你 1.2 的实现直接返回 true；③ 最实用的一条——**被打乱顺序的维度如果都是 size-1，结果仍连续**，因为 1.2 里 size-1 维的 stride 是 don't-care。例：(1,3,4) strides (12,4,1) 做 permute(1,0,2) → (3,1,4) strides (4,12,1)，跳过 size-1 维后剩下 (4,1) 仍满足紧凑降序 → 连续。
- 校验：什么样的 `order` 是合法的？长度对就行吗？（想想 (0,0,1)、(0,1,5) 这两种输入）
不是，长度对等是一个，还需要Order是有意义的。shape有n个维度，order就应该从0---n-1都有。
> ✅ 对。形式化三条：长度 == ndim；每个值 < ndim；无重复。（长度 == n 的前提下，后两条 ⟺ "0..n-1 各出现一次"，即一个排列。）(0,0,1) 死在重复，(0,1,5) 死在越界。
- 连续调两次 permute 能回到原样吗？两次的 order 是什么关系？
可以，做一次排列和对应的逆排列即可
> ✅。补一个组合律：permute(p) 后再 permute(q) 等价于一次 permute，order 为 `p[q[i]]`。取 q = p⁻¹ 时组合结果是恒等排列 (0,1,…,n-1)，即回到原样。

---

## 自测清单（能回答才算读懂）

- [x] 写出 permute(2,0,1) 作用于 (3,4,5) strides (20,5,1) 得到新 meta 的全过程。
(3,4,5) -> (5, 3, 4) -> (1, 20, 5), 此时新的meta为：(5,3,4,) (1,20,5)
> ✅ 正确。中间一步写全就是：对 i = 0,1,2 各代一次 new_shape[i] = shape[order[i]]、new_strides[i] = strides[order[i]]。
- [x] order 的方向约定是什么？写反了测试能抓住吗？为什么 (1,0) 抓不住？
不知道诶
> 补充：约定是 **new_shape\[i\] = shape\[order\[i\]]**（新第 i 维 ← 原第 order[i] 维）。写反即 new_shape\[order\[i\]] = shape\[i\]，相当于用了**逆排列**。测试 order=(2,0,1)：正方向得 (5,3,4) 与 torch 一致；反方向得 (4,5,3)，断言立刻抓住。而 (1,0) 是**自逆**排列（它的逆就是它自己，swap 两次回到自身），两个方向算出同样结果——所以只靠 (1,0) 脑补验证区分不出方向错误，这正是指南提醒"别只靠 (1,0) 验证"的原因。
- [x] 矩阵 (m,n) strides (n,1) 做 permute(1,0)（转置）后 strides 是什么？连续吗？
转置后的stride是(1,n)，不连续
> ✅ 对（m,n 都 > 1 时）。边界情况：m==1 或 n==1 时转置后仍连续——size-1 维 don't-care，(1,n) 里跳过一个后剩下的 stride==1 即满足。
- [x] 合法 order 需要满足哪三个条件？你的代码拦得住非法输入吗？
长度要与shape.size()一样。内容物要有从0到n-1所有量。其他我不知道诶
> ✅ 你这两条已覆盖三条件（见上文形式化）。剩下的一半问题是"代码怎么拦"：开一个 `std::vector<bool> seen(ndim, false)`，遍历 order——`v >= ndim` → 越界报错；`seen[v]` 已是 true → 重复报错；否则置 `seen[v] = true`。O(n) 一趟搞定，不用排序也不用集合。
- [x] 为什么 debug() 打印 permute 结果和 torch 一致，尽管物理内存一个字节都没动？
> 答案：debug() 最终走 print_data（tensor.cpp:106-121），它**按 strides 递归寻址**——逻辑下标 (i,j,k) 先翻译成物理位移 i·s₀ + j·s₁ + k·s₂ 再去读。permute 后 shape/strides 变了，但新 meta 描述的"逻辑→物理"映射恰好和 torch 读到的是同一组元素。内存布局长什么样无所谓，只要 meta 自洽，打印就正确。这就是"零拷贝 + stride 寻址"的全部意义。

---

## 实现提示

骨架（细节自己填，正文约 10 行）：

1. `CHECK_ARGUMENT(order.size() == ndim, ...)`；
2. （推荐）校验 order 是 0..ndim-1 的一个**排列**：无重复、不越界。简单做法：开一个 `std::vector<bool> seen(ndim)` 逐个标记；
3. 循环按约定方向填 `new_shape[i]` / `new_strides[i]`；
4. `return std::shared_ptr<Tensor>(new Tensor(TensorMeta{dtype, new_shape, new_strides}, _storage, _offset));`

**反面教材（不要这么写）：**

- 只重排 shape 不动 strides —— 两个数组必须按**同一个** order 重排；
- 搞反 order 的方向约定 —— 用测试用例 (2,0,1)→(5,3,4) 验证；
- 重新计算紧凑 strides —— 那是 view 的做法；permute 的意义恰恰在于保留原布局、只换维度顺序；
- `new Tensor(_meta, _storage)` 原样返回占位忘删。

---

## 验证步骤

```bash
xmake --root && xmake install --root
cd test && python test_tensor.py
```

**预期现象**：load、view、permute 三节全过（permute 打印 shape (5,3,4)、strides (1,20,5) 的逻辑数据），然后在 `===Test slice===` 抛 `TO_BE_IMPLEMENTED`——这就是 1.4 完成的标志。

---

## 学习建议

- 这个任务是纯"排列"操作，唯一真正的坑是 order 的方向约定。
- 做完后顺手验证一个直觉：`permute(2,0,1)` 的结果 `isContiguous()` 应该返回 false——如果你的 1.2 实现在这里给出 true，说明哪里串了。
