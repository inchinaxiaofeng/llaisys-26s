#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_index, size_t row_numel) {
    // 纯搬运算子：dtype 只用来算行字节数，不做任何数值转换
    const size_t row_bytes = row_numel * llaisys::utils::dsize(type);
    const int64_t *idx = reinterpret_cast<const int64_t *>(index);
    for (size_t i = 0; i < num_index; i++) {
        // 每一行是一段连续字节，整行 memcpy
        std::memcpy(out + i * row_bytes, weight + static_cast<size_t>(idx[i]) * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
