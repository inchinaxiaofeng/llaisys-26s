#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(q->ndim() == 3, "SelfAttention: q must be a 3-D tensor of shape [qlen, nh, hd].");
    CHECK_ARGUMENT(k->ndim() == 3, "SelfAttention: k must be a 3-D tensor of shape [kvlen, nkvh, hd].");
    CHECK_ARGUMENT(v->ndim() == 3, "SelfAttention: v must be a 3-D tensor of shape [kvlen, nkvh, dv].");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "SelfAttention: attn_val must be a 3-D tensor of shape [qlen, nh, dv].");
    const size_t qlen = q->shape()[0], nh = q->shape()[1], hd = q->shape()[2];
    const size_t kvlen = k->shape()[0], nkvh = k->shape()[1], dv = v->shape()[2];
    CHECK_ARGUMENT(k->shape()[2] == hd, "SelfAttention: q and k must have the same head_dim.");
    CHECK_ARGUMENT(v->shape()[0] == kvlen && v->shape()[1] == nkvh,
                   "SelfAttention: k and v must have the same kvlen and nkvhead.");
    CHECK_ARGUMENT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == dv,
                   "SelfAttention: attn_val shape must be [qlen, nh, dv].");
    CHECK_ARGUMENT(nh % nkvh == 0, "SelfAttention: nhead must be a multiple of nkvhead.");
    CHECK_ARGUMENT(kvlen >= qlen, "SelfAttention: kvlen must be no less than qlen.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   q->dtype(), qlen, kvlen, nh, nkvh, hd, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   q->dtype(), qlen, kvlen, nh, nkvh, hd, dv, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                      q->dtype(), qlen, kvlen, nh, nkvh, hd, dv, scale,
                                      llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
