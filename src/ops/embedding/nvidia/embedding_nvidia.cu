#include "embedding_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

namespace {
// 纯搬运算子：每个 block 负责一行，dtype 只用于算行字节数
__global__ void embedding_kernel(std::byte *out, const int64_t *index, const std::byte *weight, size_t row_bytes) {
    const std::byte *src = weight + static_cast<size_t>(index[blockIdx.x]) * row_bytes;
    std::byte *dst = out + static_cast<size_t>(blockIdx.x) * row_bytes;
    for (size_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_index, size_t row_numel, llaisysStream_t stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const size_t row_bytes = row_numel * llaisys::utils::dsize(type);
    embedding_kernel<<<static_cast<unsigned int>(num_index), 256, 0, s>>>(
        out, reinterpret_cast<const int64_t *>(index), weight, row_bytes);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
