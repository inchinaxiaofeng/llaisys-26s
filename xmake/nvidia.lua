-- NVIDIA CUDA 支持（仅在 xmake f --nv-gpu=y 时由 xmake.lua includes）

-- CUDA SDK 路径探测：xmake 配置 > 环境变量 > 默认路径
local cuda_dir = get_config("cuda") or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"

-- 覆盖常见卡型；bf16 只在 kernel 内做 load/store 转换（纯位运算），计算全走 float，
-- 不依赖 sm_80 的 bf16 硬件指令，因此 sm_70 起步即可
local cuda_arches = {"sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"}

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cuflags("-Xcompiler -fPIC")
    add_cugencodes(table.unpack(cuda_arches))
    -- cudart 的链接要沿依赖链传播到最终的 libllaisys.so
    add_linkdirs(path.join(cuda_dir, "lib64"), {public = true})
    add_links("cudart", {public = true})
    add_files("../src/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cuflags("-Xcompiler -fPIC")
    add_cugencodes(table.unpack(cuda_arches))
    add_files("../src/ops/*/nvidia/*.cu")
    on_install(function (target) end)
target_end()
