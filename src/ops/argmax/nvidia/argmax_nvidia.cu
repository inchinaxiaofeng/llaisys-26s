#include "argmax_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
// 单 block 归约：严格大于才更新 + 平局取下标较小者，语义与 CPU 版一致
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float s_val[256];
    __shared__ int64_t s_idx[256];
    const size_t tid = threadIdx.x;

    float best = -INFINITY;
    size_t best_i = numel; // 哨兵：本线程未扫到任何元素
    for (size_t i = tid; i < numel; i += blockDim.x) {
        float v = llaisys::ops::nvidia::to_f32(vals[i]);
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = static_cast<int64_t>(best_i);
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<size_t>(stride)) {
            float ov = s_val[tid + stride];
            int64_t oi = s_idx[tid + stride];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = s_idx[0];
        max_val[0] = vals[s_idx[0]]; // 直接拷原始位模式，不做 float 往返
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel,
            llaisysStream_t stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<1, 256, 0, s>>>(idx_ptr, reinterpret_cast<float *>(max_val),
                                               reinterpret_cast<const float *>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<1, 256, 0, s>>>(idx_ptr, reinterpret_cast<__half *>(max_val),
                                                reinterpret_cast<const __half *>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<1, 256, 0, s>>>(idx_ptr, reinterpret_cast<__nv_bfloat16 *>(max_val),
                                                       reinterpret_cast<const __nv_bfloat16 *>(vals), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
