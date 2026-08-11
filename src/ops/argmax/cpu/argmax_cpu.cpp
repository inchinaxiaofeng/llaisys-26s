#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // 首元素初始化，避免对全负数/极端输入出错
    size_t best = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < numel; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        // 严格大于才更新：并列最大时保留下标较小的（第一个）
        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(best);
    // 直接拷原始位模式，不做 float 往返转换
    max_val[0] = vals[best];
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
