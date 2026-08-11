#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                     size_t hd, size_t dv, float scale) {
    // GQA：每 nh/nkvh 个 q 头共享同一个 kv 头
    const size_t group = nh / nkvh;
    // 第 i 个 query 的绝对位置为 (kvlen - qlen) + i，只允许看到 j <= 该位置的 key
    const size_t past_len = kvlen - qlen;
    std::vector<float> score(kvlen);

    for (size_t i = 0; i < qlen; i++) {
        const size_t allowed = past_len + i + 1; // 可参与的 key 数量
        for (size_t h = 0; h < nh; h++) {
            const size_t kvh = h / group;
            const T *q_vec = q + (i * nh + h) * hd;

            // 1. 注意力分数 s_j = scale * <q, k_j>，float 累加
            float max_score = -INFINITY;
            for (size_t j = 0; j < allowed; j++) {
                const T *k_vec = k + (j * nkvh + kvh) * hd;
                float s = 0.f;
                for (size_t d = 0; d < hd; d++) {
                    s += llaisys::utils::cast<float>(q_vec[d]) * llaisys::utils::cast<float>(k_vec[d]);
                }
                s *= scale;
                score[j] = s;
                if (s > max_score) {
                    max_score = s;
                }
            }

            // 2. 数值稳定 softmax（减去最大值）
            float sum = 0.f;
            for (size_t j = 0; j < allowed; j++) {
                score[j] = std::exp(score[j] - max_score);
                sum += score[j];
            }
            const float inv_sum = 1.f / sum;

            // 3. 加权求和 out = sum_j w_j * v_j
            T *out_vec = attn_val + (i * nh + h) * dv;
            for (size_t d = 0; d < dv; d++) {
                float acc = 0.f;
                for (size_t j = 0; j < allowed; j++) {
                    acc += score[j] * llaisys::utils::cast<float>(v[(j * nkvh + kvh) * dv + d]);
                }
                out_vec[d] = llaisys::utils::cast<T>(acc * inv_sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                    size_t hd, size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               qlen, kvlen, nh, nkvh, hd, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
