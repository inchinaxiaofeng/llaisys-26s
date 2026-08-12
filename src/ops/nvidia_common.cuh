#pragma once

// NVIDIA kernel 公共工具：设备端类型转换与错误检查
// 策略与 CPU 版一致：所有计算/累加在 float 里做，半精度仅在 load/store 时转换

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#define LLAISYS_CUDA_CHECK(call)                                                            \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) \
                                     + " (" __FILE__ ":" + std::to_string(__LINE__) + ")"); \
        }                                                                                   \
    } while (0)

namespace llaisys::ops::nvidia {

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

} // namespace llaisys::ops::nvidia
