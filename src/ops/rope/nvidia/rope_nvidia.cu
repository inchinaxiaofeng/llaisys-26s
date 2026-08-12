#include "rope_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
// half-split（GPT-NeoX 式）：a = 前 d/2，b = 后 d/2；线程对应 (s, h, j)，j ∈ [0, d/2)
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    const size_t hd2 = head_dim / 2;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seqlen * nhead * hd2) {
        return;
    }
    const size_t j = idx % hd2;
    const size_t h = (idx / hd2) % nhead;
    const size_t s = idx / (hd2 * nhead);

    const float phi = static_cast<float>(pos_ids[s])
                      / powf(theta, 2.f * static_cast<float>(j) / static_cast<float>(head_dim));
    const float c = cosf(phi);
    const float sn = sinf(phi);

    const T *x = in + (s * nhead + h) * head_dim;
    T *o = out + (s * nhead + h) * head_dim;
    const float a = llaisys::ops::nvidia::to_f32(x[j]);
    const float b = llaisys::ops::nvidia::to_f32(x[j + hd2]);
    o[j] = llaisys::ops::nvidia::from_f32<T>(a * c - b * sn);
    o[j + hd2] = llaisys::ops::nvidia::from_f32<T>(b * c + a * sn);
}
} // namespace

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim, float theta,
          llaisysStream_t stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const size_t total = seqlen * nhead * (head_dim / 2);
    const unsigned int grid = static_cast<unsigned int>((total + 255) / 256);
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<grid, 256, 0, st>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                                 pos, seqlen, nhead, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<grid, 256, 0, st>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                                                  pos, seqlen, nhead, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<grid, 256, 0, st>>>(reinterpret_cast<__nv_bfloat16 *>(out),
                                                         reinterpret_cast<const __nv_bfloat16 *>(in),
                                                         pos, seqlen, nhead, head_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
