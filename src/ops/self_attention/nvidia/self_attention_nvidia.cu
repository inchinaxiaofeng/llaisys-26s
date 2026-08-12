#include "self_attention_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
// 每线程一个 (h, t)：朴素三遍（max → sum → 加权和），float 全程累加
// causal mask：q 位置 t 可见 kv 位置 s <= t + (kvlen - qlen)；GQA：kv head = h / (nh/nkvh)
// 语义与 CPU 版一致。dv <= 128 由 launcher 校验（本课程模型 dh = 128）
template <typename T>
__global__ void self_attention_kernel(T *attn, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                                      size_t hd, size_t dv, float scale) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nh * qlen) {
        return;
    }
    const size_t h = idx / qlen;
    const size_t t = idx % qlen;
    const size_t kvh = h / (nh / nkvh);
    const size_t visible = t + (kvlen - qlen) + 1; // s ∈ [0, visible)

    const T *qt = q + (t * nh + h) * hd;
    const T *kbase = k + kvh * hd;
    const T *vbase = v + kvh * dv;
    const size_t kstride = nkvh * hd;
    const size_t vstride = nkvh * dv;

    // 第一遍：scores 的 max（减 max 是 softmax 的数值稳定手段）
    float m = -INFINITY;
    for (size_t s = 0; s < visible; s++) {
        const T *ks = kbase + s * kstride;
        float dot = 0.f;
        for (size_t j = 0; j < hd; j++) {
            dot += llaisys::ops::nvidia::to_f32(qt[j]) * llaisys::ops::nvidia::to_f32(ks[j]);
        }
        dot *= scale;
        if (dot > m) {
            m = dot;
        }
    }

    // 第二遍：exp 的归一化系数
    float l = 0.f;
    for (size_t s = 0; s < visible; s++) {
        const T *ks = kbase + s * kstride;
        float dot = 0.f;
        for (size_t j = 0; j < hd; j++) {
            dot += llaisys::ops::nvidia::to_f32(qt[j]) * llaisys::ops::nvidia::to_f32(ks[j]);
        }
        l += expf(dot * scale - m);
    }

    // 第三遍：out = Σ_s p_s · v_s（score 重算，省去每线程 S 个 float 的 scratch）
    float acc[128];
    for (size_t j = 0; j < dv; j++) {
        acc[j] = 0.f;
    }
    for (size_t s = 0; s < visible; s++) {
        const T *ks = kbase + s * kstride;
        float dot = 0.f;
        for (size_t j = 0; j < hd; j++) {
            dot += llaisys::ops::nvidia::to_f32(qt[j]) * llaisys::ops::nvidia::to_f32(ks[j]);
        }
        const float p = expf(dot * scale - m);
        const T *vs = vbase + s * vstride;
        for (size_t j = 0; j < dv; j++) {
            acc[j] += p * llaisys::ops::nvidia::to_f32(vs[j]);
        }
    }

    T *o = attn + (t * nh + h) * dv;
    for (size_t j = 0; j < dv; j++) {
        o[j] = llaisys::ops::nvidia::from_f32<T>(acc[j] / l);
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                    size_t hd, size_t dv, float scale, llaisysStream_t stream) {
    if (dv > 128) {
        throw std::invalid_argument("SelfAttention(nvidia): dv > 128 is not supported yet.");
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const size_t total = nh * qlen;
    const unsigned int grid = static_cast<unsigned int>((total + 255) / 256);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<grid, 256, 0, st>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v), qlen, kvlen, nh, nkvh, hd, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<__half><<<grid, 256, 0, st>>>(
            reinterpret_cast<__half *>(attn_val), reinterpret_cast<const __half *>(q), reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v), qlen, kvlen, nh, nkvh, hd, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<__nv_bfloat16><<<grid, 256, 0, st>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val), reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k), reinterpret_cast<const __nv_bfloat16 *>(v),
            qlen, kvlen, nh, nkvh, hd, dv, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
