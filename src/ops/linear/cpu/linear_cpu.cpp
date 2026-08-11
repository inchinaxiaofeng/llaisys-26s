#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; m++) {
        const T *x_row = in + m * K;
        for (size_t n = 0; n < N; n++) {
            // W 未转置，第 n 行与 x 第 m 行都是连续 K 个元素，直接做点积
            const T *w_row = weight + n * K;
            // 统一用 float 累加，避免 f16/bf16 在长点积下精度不足
            float acc = 0.f;
            for (size_t k = 0; k < K; k++) {
                acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_row[k]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[n]);
            }
            out[m * N + n] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias),
                       M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias),
                       M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
