#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: in must be a 2-D tensor.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be a 2-D tensor.");
    CHECK_ARGUMENT(out->ndim() == 2 && out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0],
                   "Linear: out shape must be (in.shape[0], weight.shape[0]).");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: in.shape[1] must equal weight.shape[1].");
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0],
                       "Linear: bias must be a 1-D tensor of length weight.shape[0].");
        CHECK_ARGUMENT(bias->dtype() == out->dtype(), "Linear: bias must have the same dtype as out.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous.");

    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t N = weight->shape()[0];
    const std::byte *bias_data = bias == nullptr ? nullptr : bias->data();

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, N, K);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, N, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, N, K,
                              llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
