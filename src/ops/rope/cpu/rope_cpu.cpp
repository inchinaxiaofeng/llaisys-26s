#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    const size_t half = head_dim / 2;
    // 预计算每个频率分量的 freq[j] = theta^(2j/d)，角度用 pos / freq 计算。
    // 与 PyTorch 参考实现保持相同的运算顺序（正指数 + 除法），
    // 否则位置 id 较大时 float32 的角度误差会被 sin/cos 放大到超出容差。
    std::vector<float> freq(half);
    for (size_t j = 0; j < half; j++) {
        freq[j] = std::pow(theta, 2.f * static_cast<float>(j) / static_cast<float>(head_dim));
    }

    for (size_t s = 0; s < seqlen; s++) {
        const float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < nhead; h++) {
            const T *x = in + (s * nhead + h) * head_dim;
            T *y = out + (s * nhead + h) * head_dim;
            for (size_t j = 0; j < half; j++) {
                // a 为前半段，b 为后半段
                const float a = llaisys::utils::cast<float>(x[j]);
                const float b = llaisys::utils::cast<float>(x[j + half]);
                const float phi = pos / freq[j];
                const float cos_phi = std::cos(phi);
                const float sin_phi = std::sin(phi);
                y[j] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                y[j + half] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim, float theta) {
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos,
                     seqlen, nhead, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos,
                     seqlen, nhead, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos,
                     seqlen, nhead, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
