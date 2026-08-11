#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(in->ndim() == 3, "Rope: in must be a 3-D tensor of shape [seqlen, nhead, head_dim].");
    CHECK_ARGUMENT(out->ndim() == 3 && out->shape() == in->shape() && out->dtype() == in->dtype(),
                   "Rope: out must have the same shape and dtype as in.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "Rope: head_dim must be even.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->dtype() == LLAISYS_DTYPE_I64
                       && pos_ids->shape()[0] == in->shape()[0],
                   "Rope: pos_ids must be a 1-D Int64 tensor of length seqlen.");
    CHECK_ARGUMENT(theta > 0.f, "Rope: theta must be positive.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");

    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t head_dim = in->shape()[2];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), seqlen, nhead, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), seqlen, nhead, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
