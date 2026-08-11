#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1 && index->dtype() == LLAISYS_DTYPE_I64,
                   "Embedding: index must be a 1-D Int64 tensor.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be a 2-D tensor.");
    CHECK_ARGUMENT(out->ndim() == 2 && out->shape()[0] == index->numel() && out->shape()[1] == weight->shape()[1],
                   "Embedding: out shape must be (index.numel(), weight.shape[1]).");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "Embedding: out and weight must have the same dtype.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              weight->dtype(), index->numel(), weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              weight->dtype(), index->numel(), weight->shape()[1]);
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
