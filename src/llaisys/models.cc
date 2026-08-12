#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2.hpp"

#include <memory>
#include <vector>

// opaque struct：C 侧只摸指针，内部持有 C++ 模型与全部包装对象
struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> model;

    // weights() 返回的结构体本体，以及它引用的指针数组的宿主
    LlaisysQwen2Weights weights{};
    std::vector<std::unique_ptr<LlaisysTensor>> wraps; // 所有 LlaisysTensor 包装对象的宿主
    std::vector<llaisysTensor_t> attn_norm_w_ptrs;
    std::vector<llaisysTensor_t> attn_q_w_ptrs, attn_q_b_ptrs;
    std::vector<llaisysTensor_t> attn_k_w_ptrs, attn_k_b_ptrs;
    std::vector<llaisysTensor_t> attn_v_w_ptrs, attn_v_b_ptrs;
    std::vector<llaisysTensor_t> attn_o_w_ptrs;
    std::vector<llaisysTensor_t> mlp_norm_w_ptrs;
    std::vector<llaisysTensor_t> mlp_gate_w_ptrs, mlp_up_w_ptrs, mlp_down_w_ptrs;

    // 为单个 tensor_t 建包装对象，返回稳定的 llaisysTensor_t
    llaisysTensor_t wrap(llaisys::tensor_t t) {
        wraps.push_back(std::make_unique<LlaisysTensor>(LlaisysTensor{t}));
        return wraps.back().get();
    }

    void wrap_layer(const std::vector<llaisys::tensor_t> &src, std::vector<llaisysTensor_t> &dst) {
        dst.reserve(src.size());
        for (const auto &t : src) {
            dst.push_back(wrap(t));
        }
    }
};

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
                                                      int *device_ids, int ndevice) {
        int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;

        auto *m = new LlaisysQwen2Model();
        m->model = std::make_unique<llaisys::models::Qwen2Model>(meta, device, device_id);

        // 装配 weights 结构体：包装对象由 m 持有，结构体只存指针
        const auto &w = m->model->weights();
        m->weights.in_embed = m->wrap(w.in_embed);
        m->weights.out_embed = m->wrap(w.out_embed);
        m->weights.out_norm_w = m->wrap(w.out_norm_w);
        m->wrap_layer(w.attn_norm_w, m->attn_norm_w_ptrs);
        m->wrap_layer(w.attn_q_w, m->attn_q_w_ptrs);
        m->wrap_layer(w.attn_q_b, m->attn_q_b_ptrs);
        m->wrap_layer(w.attn_k_w, m->attn_k_w_ptrs);
        m->wrap_layer(w.attn_k_b, m->attn_k_b_ptrs);
        m->wrap_layer(w.attn_v_w, m->attn_v_w_ptrs);
        m->wrap_layer(w.attn_v_b, m->attn_v_b_ptrs);
        m->wrap_layer(w.attn_o_w, m->attn_o_w_ptrs);
        m->wrap_layer(w.mlp_norm_w, m->mlp_norm_w_ptrs);
        m->wrap_layer(w.mlp_gate_w, m->mlp_gate_w_ptrs);
        m->wrap_layer(w.mlp_up_w, m->mlp_up_w_ptrs);
        m->wrap_layer(w.mlp_down_w, m->mlp_down_w_ptrs);

        m->weights.attn_norm_w = m->attn_norm_w_ptrs.data();
        m->weights.attn_q_w = m->attn_q_w_ptrs.data();
        m->weights.attn_q_b = m->attn_q_b_ptrs.data();
        m->weights.attn_k_w = m->attn_k_w_ptrs.data();
        m->weights.attn_k_b = m->attn_k_b_ptrs.data();
        m->weights.attn_v_w = m->attn_v_w_ptrs.data();
        m->weights.attn_v_b = m->attn_v_b_ptrs.data();
        m->weights.attn_o_w = m->attn_o_w_ptrs.data();
        m->weights.mlp_norm_w = m->mlp_norm_w_ptrs.data();
        m->weights.mlp_gate_w = m->mlp_gate_w_ptrs.data();
        m->weights.mlp_up_w = m->mlp_up_w_ptrs.data();
        m->weights.mlp_down_w = m->mlp_down_w_ptrs.data();

        return m;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return &model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}
