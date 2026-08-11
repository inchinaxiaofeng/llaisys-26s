#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t num_row, size_t row_len, float eps) {
    for (size_t r = 0; r < num_row; r++) {
        const T *x_row = in + r * row_len;
        T *y_row = out + r * row_len;
        // 平方和用 float 累加，避免 f16/bf16 溢出或精度不足
        float sq_sum = 0.f;
        for (size_t i = 0; i < row_len; i++) {
            float v = llaisys::utils::cast<float>(x_row[i]);
            sq_sum += v * v;
        }
        const float scale = 1.f / std::sqrt(sq_sum / static_cast<float>(row_len) + eps);
        for (size_t i = 0; i < row_len; i++) {
            y_row[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(weight[i])
                                               * llaisys::utils::cast<float>(x_row[i]) * scale);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t num_row, size_t row_len, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), num_row, row_len, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), num_row, row_len, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), num_row, row_len, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
