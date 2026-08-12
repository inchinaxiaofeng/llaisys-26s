#include "swiglu_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        // 与 CPU 版一致：sigmoid 部分在 float 里算，up 做乘数
        float g = llaisys::ops::nvidia::to_f32(gate[i]);
        float u = llaisys::ops::nvidia::to_f32(up[i]);
        out[i] = llaisys::ops::nvidia::from_f32<T>(u * g / (1.f + expf(-g)));
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel,
            llaisysStream_t stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const unsigned int grid = static_cast<unsigned int>((numel + 255) / 256);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<grid, 256, 0, s>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                                                  reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<grid, 256, 0, s>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(gate),
                                                   reinterpret_cast<const __half *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<grid, 256, 0, s>>>(reinterpret_cast<__nv_bfloat16 *>(out),
                                                          reinterpret_cast<const __nv_bfloat16 *>(gate),
                                                          reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
