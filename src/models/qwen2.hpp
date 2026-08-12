#pragma once

#include "llaisys/models/qwen2.h"

#include "../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

// 权重张量的纯 C++ 视图（C API 的 LlaisysQwen2Weights 由 models.cc 负责装配）
struct Qwen2WeightsRef {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b, attn_o_w;
    std::vector<tensor_t> mlp_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w;
};

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;
    size_t _past_len = 0; // KV cache 已写入的 token 数（状态在模型内）

    Qwen2WeightsRef _w;
    std::vector<tensor_t> _kcache, _vcache; // 每层 [maxseq, nkvh, dh]

    // 单层前向，就地累加残差到 hidden 并返回之
    tensor_t _layer_forward(size_t layer, tensor_t hidden, tensor_t pos_ids, size_t past);

public:
    Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    const Qwen2WeightsRef &weights() const { return _w; }
    int64_t infer(const int64_t *token_ids, size_t ntoken);
};

} // namespace llaisys::models
