#include "linear_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
// 每线程一个 out[i,j]：Y = xW^T + b，w 的第 j 行是第 j 个输出神经元的权重（连续访存），float 累加
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *w, const T *bias, size_t M, size_t N, size_t K) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j >= N || i >= M) {
        return;
    }
    float acc = bias == nullptr ? 0.f : llaisys::ops::nvidia::to_f32(bias[j]);
    const T *x = in + i * K;
    const T *wr = w + j * K;
    for (size_t p = 0; p < K; p++) {
        acc += llaisys::ops::nvidia::to_f32(x[p]) * llaisys::ops::nvidia::to_f32(wr[p]);
    }
    out[i * N + j] = llaisys::ops::nvidia::from_f32<T>(acc);
}
} // namespace

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K, llaisysStream_t stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned int>((N + 255) / 256), static_cast<unsigned int>(M));
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<float><<<grid, 256, 0, s>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                                  reinterpret_cast<const float *>(weight),
                                                  reinterpret_cast<const float *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel<__half><<<grid, 256, 0, s>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                                                   reinterpret_cast<const __half *>(weight),
                                                   reinterpret_cast<const __half *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<__nv_bfloat16><<<grid, 256, 0, s>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight), reinterpret_cast<const __nv_bfloat16 *>(bias), M, N, K);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
