#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_index, size_t row_numel, llaisysStream_t stream);
}
