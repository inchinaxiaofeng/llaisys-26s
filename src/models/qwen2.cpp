#include "qwen2.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_id)
    : _meta(*meta), _device_type(device), _device_id(device_id) {
    const size_t nl = _meta.nlayer;
    auto dt = _meta.dtype;

    auto mk = [&](std::initializer_list<size_t> shape) {
        return Tensor::create(shape, dt, _device_type, _device_id);
    };

    // 全局权重
    _w.in_embed = mk({_meta.voc, _meta.hs});
    _w.out_embed = mk({_meta.voc, _meta.hs});
    _w.out_norm_w = mk({_meta.hs});

    // 逐层权重
    for (size_t i = 0; i < nl; i++) {
        _w.attn_norm_w.push_back(mk({_meta.hs}));
        _w.attn_q_w.push_back(mk({_meta.nh * _meta.dh, _meta.hs}));
        _w.attn_q_b.push_back(mk({_meta.nh * _meta.dh}));
        _w.attn_k_w.push_back(mk({_meta.nkvh * _meta.dh, _meta.hs}));
        _w.attn_k_b.push_back(mk({_meta.nkvh * _meta.dh}));
        _w.attn_v_w.push_back(mk({_meta.nkvh * _meta.dh, _meta.hs}));
        _w.attn_v_b.push_back(mk({_meta.nkvh * _meta.dh}));
        _w.attn_o_w.push_back(mk({_meta.hs, _meta.nh * _meta.dh}));
        _w.mlp_norm_w.push_back(mk({_meta.hs}));
        _w.mlp_gate_w.push_back(mk({_meta.di, _meta.hs}));
        _w.mlp_up_w.push_back(mk({_meta.di, _meta.hs}));
        _w.mlp_down_w.push_back(mk({_meta.hs, _meta.di}));

        _kcache.push_back(mk({_meta.maxseq, _meta.nkvh, _meta.dh}));
        _vcache.push_back(mk({_meta.maxseq, _meta.nkvh, _meta.dh}));
    }
}

tensor_t Qwen2Model::_layer_forward(size_t layer, tensor_t hidden, tensor_t pos_ids, size_t past) {
    const auto &m = _meta;
    const size_t s = hidden->shape()[0];
    const auto dt = m.dtype;

    auto mk = [&](std::initializer_list<size_t> shape) {
        return Tensor::create(shape, dt, _device_type, _device_id);
    };

    // ---------- attention ----------
    auto normed = mk({s, m.hs});
    ops::rms_norm(normed, hidden, _w.attn_norm_w[layer], m.epsilon);

    auto q = mk({s, m.nh * m.dh});
    auto k = mk({s, m.nkvh * m.dh});
    auto v = mk({s, m.nkvh * m.dh});
    ops::linear(q, normed, _w.attn_q_w[layer], _w.attn_q_b[layer]);
    ops::linear(k, normed, _w.attn_k_w[layer], _w.attn_k_b[layer]);
    ops::linear(v, normed, _w.attn_v_w[layer], _w.attn_v_b[layer]);

    auto q3 = q->view({s, m.nh, m.dh});
    auto k3 = k->view({s, m.nkvh, m.dh});
    auto v3 = v->view({s, m.nkvh, m.dh});

    auto qr = mk({s, m.nh, m.dh});
    auto kr = mk({s, m.nkvh, m.dh});
    ops::rope(qr, q3, pos_ids, m.theta);
    ops::rope(kr, k3, pos_ids, m.theta);

    // 写入本层 KV cache
    core::context().setDevice(_device_type, _device_id);
    auto *api = core::context().runtime().api();
    const size_t kv_bytes = s * m.nkvh * m.dh * k->elementSize();
    api->memcpy_sync(_kcache[layer]->slice(0, past, past + s)->data(), kr->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    api->memcpy_sync(_vcache[layer]->slice(0, past, past + s)->data(), v3->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

    // 用完整缓存做注意力（kvlen = past + s >= s，满足 self_attention 约束）
    auto kall = _kcache[layer]->slice(0, 0, past + s);
    auto vall = _vcache[layer]->slice(0, 0, past + s);
    auto attn = mk({s, m.nh, m.dh});
    const float scale = 1.f / std::sqrt(static_cast<float>(m.dh));
    ops::self_attention(attn, qr, kall, vall, scale);

    auto o = mk({s, m.hs});
    ops::linear(o, attn->view({s, m.hs}), _w.attn_o_w[layer], nullptr);
    ops::add(hidden, hidden, o); // 残差，就地写回

    // ---------- MLP ----------
    ops::rms_norm(normed, hidden, _w.mlp_norm_w[layer], m.epsilon);
    auto gate = mk({s, m.di});
    auto up = mk({s, m.di});
    ops::linear(gate, normed, _w.mlp_gate_w[layer], nullptr);
    ops::linear(up, normed, _w.mlp_up_w[layer], nullptr);
    auto act = mk({s, m.di});
    ops::swiglu(act, gate, up);
    auto down = mk({s, m.hs});
    ops::linear(down, act, _w.mlp_down_w[layer], nullptr);
    ops::add(hidden, hidden, down);

    return hidden;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    const auto &m = _meta;
    const size_t s = ntoken;
    const size_t past = _past_len;
    CHECK_ARGUMENT(past + s <= m.maxseq, "context length exceeds maxseq");

    auto mk = [&](std::initializer_list<size_t> shape, llaisysDataType_t dt) {
        return Tensor::create(shape, dt, _device_type, _device_id);
    };

    // token ids -> hidden states
    auto ids = mk({s}, LLAISYS_DTYPE_I64);
    ids->load(token_ids);
    auto hidden = mk({s, m.hs}, m.dtype);
    ops::embedding(hidden, ids, _w.in_embed);

    // 绝对位置 = past + [0..s)
    std::vector<int64_t> pos(s);
    for (size_t i = 0; i < s; i++) {
        pos[i] = static_cast<int64_t>(past + i);
    }
    auto pos_ids = mk({s}, LLAISYS_DTYPE_I64);
    pos_ids->load(pos.data());

    for (size_t i = 0; i < m.nlayer; i++) {
        hidden = _layer_forward(i, hidden, pos_ids, past);
    }

    // 只取最后一个 token 的 hidden 过末 norm + lm_head
    auto last = hidden->slice(0, s - 1, s);
    auto normed = mk({1, m.hs}, m.dtype);
    ops::rms_norm(normed, last, _w.out_norm_w, m.epsilon);
    auto logits = mk({1, m.voc}, m.dtype);
    ops::linear(logits, normed, _w.out_embed, nullptr);

    auto max_idx = mk({1}, LLAISYS_DTYPE_I64);
    auto max_val = mk({1}, m.dtype);
    ops::argmax(max_idx, max_val, logits->view({m.voc}));

    core::context().setDevice(_device_type, _device_id);
    int64_t next = 0;
    core::context().runtime().api()->memcpy_sync(&next, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    _past_len += s; // 更新状态（含本轮写入 cache 的 s 个 token）
    return next;
}

} // namespace llaisys::models
