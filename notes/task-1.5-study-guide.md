# 任务 1.5 学习指南：实现 `Tensor::slice`

## 任务目标

实现 `src/tensor/tensor.cpp`（268-271 行）中的：

```cpp
tensor_t slice(size_t dim, size_t start, size_t end) const;
```

语义：**沿第 dim 维切出 [start, end) 区间**，零拷贝。测试用例（test_tensor.py:41-49）：(3,4,5) 连续张量做 `slice(2, 1, 4)` → shape (3,4,3)、strides 仍是 (20,5,1)，对应 torch 的 `[:, :, 1:4]`。

三个 Meta Transform 至此凑齐，各自动了三样东西里的哪几样：

- view：shape 换掉、strides 按新 shape 重算紧凑、offset 继承；
- permute：shape/strides 按同一个 order 重排、offset 继承；
- slice：**只改 shape\[dim\]、strides 原封不动、offset 前移**。

slice 是第一个会动 `_offset` 的操作。

---

## 核心：offset 前移多少？—— 本任务唯一的真坑

先回答一个事实问题：`_offset` 的单位是什么？看 `data()`（tensor.cpp:48-54）：`_storage->memory() + _offset`，是指针算术的起点；README 也写明 offset 是 **in bytes**。而 `strides` 的单位是**元素**（print_data 里 `data[i * strides[dim]]`，data 是 `T*`）。

所以新起点 = 旧起点 + 沿 dim 维跳过 start 个元素的物理距离：

```
new_offset = _offset + start × strides[dim] × elementSize()   // 单位：字节
```

忘掉 `elementSize()` 这个换算，就是本任务最大的坑。而且测试**恰好用 i64**（8 字节），忘掉 ×8 不会崩、shape/strides 断言照过，只有 check_equal 抓得到——它是怎么抓的？去看 `test_utils.py:116-160`：check_equal 用 `right = Σ strides[i]·(shape[i]-1)` 算出从 `data_ptr()` 出发的最大可达位移，分配 `right+1` 个元素的一维 buffer，**从 data_ptr() 平铺 memcpy**，再 `torch.as_strided` 安上 shape/strides。整条链路对非连续布局完全免疫，唯一的输入就是 `data_ptr()` —— offset 错了，起点就错了，后面全错。

第二个要想通的问题：为什么 strides 一个字都不用改？slice 只是"沿某维从中途开始数、数到 end 为止"，**任何一维上相邻元素的物理间距都没有变**。变的只有三样：dim 维还剩几个元素（end-start）、从哪开始（offset）、以及逻辑下标的范围。这也正是 slice 和 view 的本质区别：view 要从布局假设里**推导**新 strides，所以要求连续；slice 只是继承旧布局，对连续性**零要求**——permute 过的非连续张量照样直接切。

---

## 阅读路线（约 15 分钟）

1. `test/test_tensor.py:41-49` — slice 节的断言：shape、strides、check_equal。注意 48 行断言的又是**原**张量（和 view/permute 节同款写法），切片本身的连续性测试不管，自己拿 torch 对。
2. `python/llaisys/tensor.py:92-97` — Python 层这次连长度断言都没有，裸转 `c_size_t` 直传。想想传负数会发生什么。
3. `src/llaisys/tensor.cc:89-95` — C API 也是直通。校验的全部责任都在 C++ 这一层。
4. `src/tensor/tensor.cpp:268-271` — stub 本体。老规矩，盯着它的 return 看。
5. `src/tensor/tensor.cpp:48-54` — `data()`：确认 offset 单位是字节。
6. `test/test_utils.py:116-160` — check_equal 的 flat memcpy + as_strided，上文已剧透，值得亲眼看一遍。

**思考问题：**

- 写出 `slice(2, 1, 4)` 作用于 (3,4,5) i64（strides (20,5,1)，原 offset 0）的完整新状态：new_shape、new_strides、new_offset（字节）。

切index为2的维度上，start为1，end为但不包含4, offset = _offset + 1•strides\[2\]*elementSize() = 8

new_shape(3,4,3), new_strides(20,5,1), new_offset:8
> ✅ 三样全对。但要命的是：笔记公式写对了，代码 `tensor.cpp:286` 却漏了 `* elementSize()`——offset 算成 1 字节而不是 8 字节，这次 check_equal 挂的就是它。公式没落进代码等于没写。

- stub 的 `return new Tensor(_meta, _storage)` 犯了几个错？（和 1.3 view 的 stub 对照）
meta没有写对啦，应该用新的meta
> 半对。stub 是**两个**错：① `_meta` 原样传（shape 没改）；② 第三个参数没传，`_offset` 丢成 0。和 1.3 view 的 stub 同款双错——以后见到 stub 的 return，先数它丢了几样东西。

- 合法入参要满足什么？`dim`、`start`、`end` 各自列条件。`start == end`（空切片，torch 是允许的）该不该放行？放行的话实现要付出什么代价？

dim不越界，start不越界，end不越界，start不大于end。
空切片应该是没有任何代价的，至少我是这么理解的
> ✅ 对，收紧一下：`start ≤ end ≤ shape[dim]` 一条链就够——end 允许**等于** shape[dim]（所以是 ≤ 不是 <）；start 的上界被 start ≤ end 蕴含，不用单独查。空切片零代价的理解也对：shape[dim]=0 → numel=0，offset 公式照常成立（即使指到该维末尾也不会有任何访存）。你的实现放行了 start == end，和这个答案一致 ✓

- Python 侧传 `slice(2, -1, 4)` 会发生什么？（提示：`c_size_t` 对负数是模 2⁶⁴ 环绕）你的边界校验拦得住吗？

Python中，c_size_t就是C的size_t，不能传负数，会传递一个超大的数进来使用。但是这个不应该出现，因为代码应当保证在任何一个平台上，行为都是一致的，且不能依赖任何UB。
尽管我不确定c_size_t的warp是不是ub，但是size_t本身就是一个根据平台位宽变化的值。
> 环绕判断对，两处澄清：① ctypes 里 `c_size_t(-1)` 的环绕是**定义良好**的行为（模 2⁶⁴），不是 UB；C++ 里有符号→无符号转换同理（同 1.3 那条 `view(-1)` 的订正：不是 UB，但依赖它是糟糕设计）。② 问题还有一半没答："你的校验拦得住吗？"——拦得住。start 环绕成 SIZE_MAX → `start > end` 抛错；负数若传在 end 上 → `end > shape[dim]` 抛错。你的两条边界校验正好各管一个。

- (3,4,5) 连续张量：`slice(0, 1, 3)` 的结果连续吗？`slice(1, 1, 3)` 呢？用你 1.2 的判据逐维过一遍，再和 torch 的 `t[1:3]` / `t[:, 1:3]` 对答案。规律是什么？

- `slice(dim, 0, shape[dim])` 等价于什么操作？

等价与单独将某一个维度展开成一个1 Dim Tensor，即"取得某一个维度"。
> ❌ 不对。区间 [0, shape[dim]) 是**整个维度**，所以 new_shape 和原 shape 一模一样、strides 不变、offset 不变——这是**恒等操作**（返回一个与 this 等价的视图）。关键：slice 永远不改变 ndim，不存在"把某维展开成 1-D"的效果；你想要的"取某一维"是 select/index 的语义（ndim − 1），这个 API 没有。

---

## 自测清单（能回答才算读懂）

- [ ] 一句话：slice 改了什么、没改什么？
- [ ] offset 忘乘 `elementSize()`，测试里哪些断言会挂、哪些照过？为什么？
- [ ] 为什么 view 必须要求原张量连续，而 slice 对连续性零要求？
- [ ] 切 dim 0 保持连续的规律，有没有例外？（想想 size-1 维和恒等切片）
- [ ] 对一个 permute 过的非连续张量直接 slice，meta 怎么变？需要先 contiguous 吗？

---

## 实现提示

骨架（细节自己填，正文约 10 行）：

1. 校验 `dim < ndim()`；
2. 校验 `start` 与 `end`：`end <= shape()[dim]`，以及 start 与 end 的大小关系（空切片放行与否，见思考问题）；
3. `std::vector<size_t> new_shape = shape();` 然后只改 `new_shape[dim]`；
4. 按核心一节的公式算 `new_offset`（注意单位换算，注意 `strides()[dim]` 是 `ptrdiff_t`）；
5. `return std::shared_ptr<Tensor>(new Tensor(TensorMeta{dtype(), new_shape, strides()}, _storage, new_offset));` —— strides 原样传。

**反面教材（不要这么写）：**

- offset 忘乘 `elementSize()` —— 字节/元素混淆，本任务唯一真正的坑；
- 照 view 的样子重算紧凑 strides —— slice 的意义恰恰在于**保留**原布局（包括"空隙"），重算紧凑 strides 描述的就不是同一块数据了；
- 只改 shape 不动 offset —— 读到的还是从头开始的数据，shape/strides 断言全过、check_equal 挂；
- 任何 memcpy / 新 Storage —— slice 必须零拷贝；
- 留着 stub 的 return 当占位忘了删。

---

## 验证步骤

```bash
xmake --root && xmake install --root
cd test && python test_tensor.py
```

**预期现象**：load、view、permute、slice 四节全过，打印 `Test passed!`——Assignment #1 至此完成。接着做 Task-1.6：commit & push，确认 GitHub Actions 绿。

---

## 学习建议

- 做完后把三个 transform 摆在一起默写那张"动了哪几样"的对照表——想通它，Assignment #1 的"零拷贝 + stride 寻址"心智模型就闭环了。
- 顺手自验连续性：你的 `slice(2,1,4)` 结果 `isContiguous()` 应该是 false（和 torch 的 `t[:, :, 1:4]` 一致）；再试 `slice(0,1,3)` 应该是 true。测试不查这两条，自己查。
- slice 出来的非连续张量，正是挑战任务 `contiguous()` 最典型的输入——先留个印象，作业二之后用得上。
