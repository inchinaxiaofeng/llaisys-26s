# 任务 1.1 学习指南：实现 `Tensor::load`

## 任务目标

实现 `src/tensor/tensor.cpp` 中的：

```cpp
void load(const void *src);
```

语义：**把主机（CPU）内存里的数据，拷贝到这个张量所在的设备内存里**。

- 现在只有 CPU，所以本质上是一次内存拷贝；但必须写成**设备无关**的形式，将来张量在 GPU 上时同一份代码也要能工作（作业 #4 会直接受益）。
- README 的核心提示："查看构造函数了解如何获取当前设备上下文的运行时 API，并执行从主机到设备的内存复制。"

---

## 架构总览：四层调用链

```
test_tensor.py                     Python 测试
    │  llaisys_tensor.load(ptr)
    ▼
python/llaisys/tensor.py           ctypes 封装（Python 风格 API）
    │  LIB_LLAISYS.tensorLoad(...)
    ▼
src/llaisys/tensor.cc              C API（__C / __export，供 ctypes 调用）
    │  tensor->tensor->load(data)
    ▼
src/tensor/tensor.cpp              C++ 核心实现  ← 你要改的地方
    │
    ▼
core::context() → Runtime → LlaisysRuntimeAPI（函数指针表）
    │
    ▼
src/device/cpu/cpu_runtime_api.cpp  具体设备的实现（CPU 版）
```

读代码时随时回到这张图定位自己在哪一层。

---

## 阅读路线（按顺序，约 1 小时）

### 第 1 层：Python 层

- `test/test_tensor.py:16` — 看 `load` 怎么被调用
- `python/llaisys/tensor.py:75-76` — `load(data: c_void_p)` 只是转发给了第 2 层的 `tensorLoad`

**思考问题：**
- `llaisys_tensor.load(torch_tensor.data_ptr())` 传入的是什么？（一个 `c_void_p` 裸指针）
回答问题：data_ptr()传递的是PyTorch张量物理数据在内存中的起始地址。
PyTorch在底层为torch_tensor分配了一块存放真正数值的内存空间，data_ptr()返回的就是这块内存的起始指针。
返回值只包含内存首地址，不包含Shape，Dtype等元数据。

这里的操作是：将Pytorch的首地址传递给了llaisys_tensor.load()，
底层C++代码拿到指针后，会根据 llaisys_tensor 自己算好的字节大小($Shape \times ElementSize$)，
通过memcpy或者cudaMemcpy把PyTorch内存里的二进制字节直接按块拷贝到自己申请的内存空间里。

✅ 补充（两个细节）：
1. `torch_tensor.data_ptr()` 在 Python 里返回的其实是一个 **int**（地址的整数值），ctypes 依据 `tensorLoad` 声明的形参类型 `c_void_p` 把它转成裸指针。
2. “按块平拷”成立有个隐含前提：**源 torch 张量必须连续（contiguous）**。测试里 `torch.arange(60).reshape(3, 4, 5)` 恰好连续，逻辑顺序 == 物理顺序；若源张量是非连续视图，data_ptr + 平拷就会按错误顺序拷数据（strides 信息并没有随指针传过去）。
- 谁负责保证 `src` 指向的内存大小足够？
C API 的底层接口都具有“信任调用者”的传统思想，校验工作需要放到更高的Python接口中。

✅ 补充：事实上这个项目里 Python 层也没做校验（`tensor.py:75-76` 直接转发）。`load` 的定位就是**最底层原语**，契约是“调用者保证 src 至少有 numel × elementSize 字节可读、且源连续”。llaisys 一侧只能算出自己**需要**多少字节，无法得知 src **实际**有多大——这正是它无法校验的根本原因。

### 第 2 层：ctypes → C API 层

- `src/llaisys/tensor.cc:68-72` — `tensorLoad` 直接转发给 C++ 的 `Tensor::load`

**思考问题：**
- 这一层几乎只是转发，存在的意义是什么？
  （提示：C ABI 才能被 ctypes 调用；C++ 有 name mangling）


这个是系统编程和跨语言调用（FFI）中的一个底层机制
C++ 支持函数重载、类和命名空间，编译器需要把同名实体区分开，于是会偷偷修改函数名（Name Mangling）：编译后动态链接库的符号表里，源码中的名字会被改写成各种内部名字。
不同编译器（甚至同一编译器的不同版本）的 Name Mangling 规则都不同，而 Python 的 ctypes 只能按字符串（"tensorLoad"）去符号表里查，不改写就会 symbol not found。
C 语言没有重载和命名空间，因此 C 编译器规则非常简单：源码里函数叫什么，符号表里就叫什么。这就是 C ABI，是操作系统和多语言交互的通用标准。

```cpp
#ifdef __cplusplus
extern "C" {
#endif

void tensorLoad(llaisysTensor_t tensor, const void *data);

#ifdef __cplusplus
}
#endif
```

这种'extern "C"'就是为了告诉C++编译器：关闭函数Name Mangling，严格按照C ABI标准导出符号。
**更好玩的是，在Rust中，与C的接口也是extern "C", 我理解应该是同样一种考量**

在这里，extern "C" 声明在 `include/llaisys.h`（12-20 行），通过宏定义为 `__C`
引入链（订正：不是直接 include）：`src/llaisys/tensor.cc` → `llaisys_tensor.hpp` → `include/llaisys/tensor.h` → `include/llaisys.h`。声明（tensor.h）和实现（tensor.cc）都包在 `__C { }` 里，编译出来的就是天然符合 C ABI 的纯净符号，然后再调用实际的 C++ 逻辑。

`python/llaisys` ---> `src/llaisys/tensor.cc` ---> `src/tensor/tensor.cpp`
Python前端          C API适配层                 C++核心逻辑

✅ 补充：除了 Name Mangling，这一层还有两层意义：
1. **C++ ABI 本身不可移植**：`std::string`、`std::vector`、`std::shared_ptr` 的内存布局没有跨编译器标准，不能跨过 FFI 边界。所以 C API 签名里只有 POD 类型（指针、size_t、enum）。注意 `LlaisysTensor` 内部其实包着一个 `std::shared_ptr<Tensor>`（`llaisys_tensor.hpp:8`），但对 Python 只暴露不透明指针 `llaisysTensor_t`（`include/llaisys/tensor.h:7`）——C++ 类型从不越界。
2. **配合 `__export`（`llaisys.h:4-10`）控制符号可见性**：Windows 上是 `__declspec(dllexport)`，GCC 上是 `__attribute__((visibility("default")))`。没有它符号根本不进动态库导出表，extern "C" 也救不了。

### 第 3 层：C++ 张量层（重点）

读 `src/tensor/tensor.cpp` 里的两段现成"教材"：

**教材一：`Tensor::create`（14-43 行）**

重点看 35-42 行：
- 如何通过 `core::context()` 拿到当前线程的上下文
- 如何 `setDevice(device_type, device)` 切换设备
- 如何向 `runtime()` 申请存储（`allocateHostStorage` / `allocateDeviceStorage`）

这就是 README 让你"查看构造函数"的原因——`load` 里获取运行时 API 的写法与这里完全一样。

**教材二：`Tensor::debug`（155-170 行）—— 本任务最重要的参考**

```cpp
core::context().setDevice(this->deviceType(), this->deviceId());   // 156 行
core::context().runtime().api()->device_synchronize();
...
core::context().runtime().api()->memcpy_sync(                      // 163-167 行
    tmp_tensor->data(),
    this->data(),
    this->numel() * this->elementSize(),
    LLAISYS_MEMCPY_D2H);
```

它做了一次 **设备→主机（D2H）** 的拷贝。你要做的 `load` 恰好是反方向：**主机→设备（H2D）**。注意它在拷贝前先 `setDevice` 到了张量自己所在的设备（156 行），想想为什么必须先做这一步。

GPU驱动采用“线程局部状态”，当前处于激活状态的设备ID与调用线程强绑定的（CUDA 里对应 cudaSetDevice 设置的 current device）。

✅ 补充（这套机制在 llaisys 里的具体实现）：
- `core::context()` 返回的是 **thread_local 的 Context 单例**（`context.cpp:74-77`）；每个 (device_type, device_id) 对应一个独立 `Runtime`，Context 里只有一根 `_current_runtime` 指针。
- `Context::setDevice`（`context.cpp:52-66`）切换 `_current_runtime`，并调 `Runtime::_activate()` → `api()->set_device(id)`（`runtime.cpp:24-27`）——CUDA 后端下这就是 cudaSetDevice。
- `context().runtime()` 会断言当前必须有已激活的 Runtime（`context.cpp:68-71`）。不调 setDevice 就拿 runtime，拿到的可能是**别的设备**的 API 表和分配器——在错误的设备上分配显存、发起拷贝。
- 注意 CPU 后端的 `setDevice` 是空操作（`cpu_runtime_api.cpp:13-15`），所以现在偷懒不写也能过测试；但 GPU 后端（作业 #4）会立刻出错。这就是从任务 1.1 就要养成习惯的原因。

### 第 4 层：设备抽象层

- `src/core/context/context.hpp` — `Context` 是每线程单例，管理所有设备的 `Runtime`
- `src/core/runtime/runtime.hpp` — `Runtime` 持有 `const LlaisysRuntimeAPI *_api`
- `src/core/storage/storage.hpp` — `Storage` 是一块内存 + 所属 Runtime
- `include/llaisys/runtime.h` — `memcpy_sync_api` 的 typedef 在 22 行，`LlaisysRuntimeAPI` 函数指针表结构体在 25-38 行，找到：
  ```cpp
  typedef void (*memcpy_sync_api)(void *, const void *, size_t, llaisysMemcpyKind_t);
  ```
- `include/llaisys.h` — 找 `llaisysMemcpyKind_t` 枚举，确认"主机到设备"的取值名称
- `src/device/cpu/cpu_runtime_api.cpp` — 看 CPU 版 `memcpy_sync` 的实际实现

---

## 自测清单（能回答才算读懂）

- [ ] 张量的数据指针怎么算？为什么 `data()`（45-51 行）不直接返回 `_storage->memory()`？
❌ 原答案（“通过 strides 获得偏移步长”）不对——strides 决定的是**元素之间的间隔**，而 `data()` 要找的是**起点**，两者是正交的两件事。

✅ 正确答案：`data()` 返回 `_storage->memory() + _offset`（`tensor.cpp:45-51`）。`_offset` 是**字节**偏移（`memory()` 返回 `std::byte *`），记录这个张量的逻辑元素 (0,0,...,0) 在共享存储中的起始位置。
- 新建张量 `_offset = 0`，二者恰好相等，看不出区别；
- 但 `slice(dim, start, end)` 会让新张量**共享同一块 Storage**，同时把 `_offset` 加上 `start × strides[dim] × elementSize()` 个字节。此时若 `data()` 不加 `_offset`，slice 出来的张量就会从 Storage 头部开始读，数据全错；
- 一句话：**offset 管“从哪里开始”，strides 管“每走一步多远”**。view/permute 只改 shape/strides，只有 slice 会动 offset（想想为什么前两个不需要动）。
- [ ] 要拷贝的字节数怎么算？用哪两个现成的成员函数？
✅ `Tensor::numel()` × `Tensor::elementSize()`，正确——`debug()` 里就是这么算的（`tensor.cpp:166`）。注意乘积单位是**字节**，正好对应 `memcpy_sync` 第三个参数的单位。
- [ ] 为什么 `load` 里要先 `setDevice(this->deviceType(), this->deviceId())`，而不是直接用当前 runtime？
      （想想：当前上下文停在设备 0，而张量在设备 1 上，会发生什么？CUDA 下这对应什么概念？）
GPU驱动采用“线程局部状态”，当前处于激活状态的设备ID与调用线程强绑定的（CUDA 的 current device，对应 cudaSetDevice）。

✅ 补充：
1. 若上下文停在设备 0 而张量在设备 1：`context().runtime()` 返回的是**设备 0** 的 Runtime——API 表和分配器全错了。CUDA 下 cudaMalloc、kernel launch、stream 操作都隐含作用于 current device，整套流程会错位。
2. 同步原语同样作用于当前设备：`debug()` 里先 setDevice 再 `device_synchronize()`（`tensor.cpp:156-157`），就是为了同步**张量自己**所在的设备。`load` 本身虽然没有 synchronize 调用，但必须先 setDevice 的理由和 debug 完全一致：让“当前 Runtime”就是张量的 Runtime。
- [ ] `load` 需要处理非连续（strides 不规则）的张量吗？测试里 `load` 在什么时候被调用？
结论“不需要”是对的，但理由不是“padding 到相同长度”。

✅ 正确理由：
1. 看测试：`load` 只在 `test_tensor.py:16` 被调用一次，对象是**刚 create 出来的张量**——`Tensor::create` 生成的 strides 永远是紧凑连续的（`tensor.cpp:27-30` 的倒序累乘），且 `_offset = 0`。view/permute/slice 都发生在 load **之后**。
2. 看契约：`load` 的语义就是“把 numel × elementSize 个字节平拷到 `data()` 开始的缓冲区”，它根本不解释 strides；源 torch 张量（arange + reshape）也是连续的，两边逻辑顺序一致。
3. 什么时候才需要关心非连续？要往**非连续视图**里 load 时，就得按 strides 做 strided 拷贝（或先 `contiguous()` 再平拷）——那是后续 `contiguous()` / `reshape()` 任务的事，不属于本任务。
- [ ] `Tensor` 和 `Storage` 为什么要分离？（提示：view/permute/slice 共享同一块存储）
这里的 `view` `permute` `slice` 都是对现有张量的“视图变换”。共同特点是：不改变底层物理内存的数据，只修改描述数据的Metadata。
1. view
* 概念：在改变张量Shape的同时，保持元素总数不变（Pytorch的Reshape/view）
* 元数据变化：更新Shape，并根据新的Shape重新计算连续的Strides
* 例如：将(12,)的一维Tensor变为(3,4)的2Dim Tensor。
2. Permute
* 概念：重新排列Tensor的维度顺序（将 Dim 0 和 Dim 1 对调）
* 元数据变化：Shape内部元素重排，同时Strides顺序也被相应打乱。
* 例子：Shape(3,4), strides(4,1)的Tensor, 做 permute(1,0)后为：Shape(4,3),strides(1,4)。但是物理内存没有任何变化
3. slice
* 概念：截取张量在某个/某些维度上的一个连续区间(tensor\[1:3, :\])
* 元数据变化：更新Shape(变小), 同时调整Offset(起点指针偏移量)，必要时调整strides(如果设置了step步长)
* 例子:截取第一行后，新Tensor的指针起点向前平移了 1 * stride\[0\] 个元素，但是指向的依然是同一块物理内存
* ✅ 注意单位：代码里 `_offset` 以**字节**存储（`tensor.cpp:46`），实际加的是 `start × strides[dim] × elementSize()` 字节。测试里的 `slice(2, 1, 4)`（i64，strides[2]=1）就是 `_offset += 1 × 1 × 8 = 8` 字节。

为什么Tensor和Storage要分离

1. 职责解耦（逻辑视图 vs 物理存储）
* Storage（物理层）：只负责物理内存的生命周期管理（代码里靠 `std::shared_ptr<Storage>` 引用计数自动实现：只要还有一个视图活着，底层内存就不会释放）
* Tensor（逻辑层）：负责把一维缓冲区解释成多维结构，持有 Storage 的智能指针，并用 Shape、Strides、offset、dtype 建立“坐标 → 地址”的映射。
2. O(1) 时间复杂度的 Zero-Copy 操作：Tensor 的维度转换、转置、切片，只改元数据不碰数据。
3. 省内存、避免拷贝：N 个视图共享同一块底层存储，不必为每个视图复制一份数据（对大模型权重共享尤其重要）。
4. 视图修改自动同步：所有视图指向同一块物理内存，对任一视图的就地（in-place）修改，其他视图立刻可见——这正是 PyTorch 里 view 的语义。

---

## 实现提示

骨架（细节自己填，正文不应超过 5 行）：

1. 把上下文切到**这个张量自己**所在的设备；
2. 通过 `core::context().runtime().api()` 拿到 API 表；
3. 调 `memcpy_sync`：目标是 `this->data()`，源是 `src`，长度是元素数 × 元素大小，方向选主机到设备。

**反面教材（不要这么写）：**
- 直接用 `std::memcpy`
- 写 `if (device == CPU)` 分支

这两种写法在纯 CPU 上都能过测试，但违背了设备抽象的设计意图，作业 #4 会返工。

---

## 验证步骤

```bash
# 重新编译并安装共享库
xmake && xmake install
```

**注意一个坑**：`test/test_tensor.py` 会把 load、view、permute、slice 串起来跑，而 `isContiguous()`（任务 1.2）还是 `TO_BE_IMPLEMENTED()`，会在 load 之后立刻抛异常。所以先写一个只测 load 的小脚本：

```python
import torch
import llaisys
from test_utils import torch_dtype, llaisys_dtype, llaisys_device, check_equal

torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
llaisys_tensor = llaisys.Tensor((3, 4, 5), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu"))
llaisys_tensor.load(torch_tensor.data_ptr())
llaisys_tensor.debug()                      # 肉眼检查打印的数据
assert check_equal(llaisys_tensor, torch_tensor)
print("load OK")
```

（`check_equal` 等工具函数在 `test/test_utils.py` 里，在 `test/` 目录下运行脚本。）

load 验证通过后再开始任务 1.2。

---

## 学习建议

- 先读、后写。这个任务代码量极小，分值全在"理解架构"上。
- 读的时候顺手画一张 `Tensor → Storage → Runtime → Context → RuntimeAPI` 的关系图，作业 #4 实现 CUDA Runtime 时会直接用到。
- 卡住时先用 `debug()` 打印中间状态，别急着搜答案。
