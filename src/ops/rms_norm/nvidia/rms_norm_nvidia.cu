#include "rms_norm_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
// 每行一个 block：第一遍归约 Σx²，第二遍缩放写回；与 CPU 版同为 float 累加
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *w, size_t row_len, float eps) {
    __shared__ float s_sum[256];
    const size_t row = blockIdx.x;
    const T *x = in + row * row_len;

    float sum = 0.f;
    for (size_t i = threadIdx.x; i < row_len; i += blockDim.x) {
        float v = llaisys::ops::nvidia::to_f32(x[i]);
        sum += v * v;
    }
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < static_cast<size_t>(stride)) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float r = rsqrtf(s_sum[0] / static_cast<float>(row_len) + eps);
    T *o = out + row * row_len;
    for (size_t i = threadIdx.x; i < row_len; i += blockDim.x) {
        o[i] = llaisys::ops::nvidia::from_f32<T>(llaisys::ops::nvidia::to_f32(x[i]) * r * llaisys::ops::nvidia::to_f32(w[i]));
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t num_row, size_t row_len, float eps, llaisysStream_t stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const unsigned int grid = static_cast<unsigned int>(num_row);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<grid, 256, 0, s>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                                    reinterpret_cast<const float *>(weight), row_len, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<grid, 256, 0, s>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                                                     reinterpret_cast<const __half *>(weight), row_len, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<grid, 256, 0, s>>>(reinterpret_cast<__nv_bfloat16 *>(out),
                                                            reinterpret_cast<const __nv_bfloat16 *>(in),
                                                            reinterpret_cast<const __nv_bfloat16 *>(weight), row_len, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
