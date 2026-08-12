#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) \
                                     + " (" __FILE__ ":" + std::to_string(__LINE__) + ")"); \
        }                                                                                    \
    } while (0)

namespace llaisys::device::nvidia {

namespace runtime_api {

static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unknown memcpy kind.");
    }
}

int getDeviceCount() {
    // Context 构造时会探测本函数；无卡/驱动异常必须返回 0 而不是抛异常，
    // 否则 nv-gpu 构建在纯 CPU 环境下会直接崩掉
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError(); // 清除错误状态，避免污染后续 CUDA 调用
        return 0;
    }
    return count;
}

void setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return static_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size)); // 锁页内存，H2D/D2H 拷贝更快
    return ptr;
}

void freeHost(void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), static_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
