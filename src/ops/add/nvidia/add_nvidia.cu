#include "add_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        c[i] = llaisys::ops::nvidia::from_f32<T>(llaisys::ops::nvidia::to_f32(a[i]) + llaisys::ops::nvidia::to_f32(b[i]));
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel,
         llaisysStream_t stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const unsigned int grid = static_cast<unsigned int>((numel + 255) / 256);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<grid, 256, 0, s>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                                               reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<grid, 256, 0, s>>>(reinterpret_cast<__half *>(c), reinterpret_cast<const __half *>(a),
                                                reinterpret_cast<const __half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<__nv_bfloat16><<<grid, 256, 0, s>>>(reinterpret_cast<__nv_bfloat16 *>(c),
                                                       reinterpret_cast<const __nv_bfloat16 *>(a),
                                                       reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
