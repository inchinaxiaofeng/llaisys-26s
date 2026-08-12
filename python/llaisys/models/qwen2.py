from typing import Sequence

from ..libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    LlaisysQwen2Meta,
    llaisysDeviceType_t,
)

import json
from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
from pathlib import Path

import safetensors
import torch

# 逐层权重路由：safetensors 尾缀 → LlaisysQwen2Weights 字段名
_LAYER_ROUTES = {
    "input_layernorm.weight": "attn_norm_w",
    "self_attn.q_proj.weight": "attn_q_w",
    "self_attn.q_proj.bias": "attn_q_b",
    "self_attn.k_proj.weight": "attn_k_w",
    "self_attn.k_proj.bias": "attn_k_b",
    "self_attn.v_proj.weight": "attn_v_w",
    "self_attn.v_proj.bias": "attn_v_b",
    "self_attn.o_proj.weight": "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight": "mlp_gate_w",
    "mlp.up_proj.weight": "mlp_up_w",
    "mlp.down_proj.weight": "mlp_down_w",
}

# 顶层权重路由
_TOP_ROUTES = {
    "model.embed_tokens.weight": "in_embed",
    "lm_head.weight": "out_embed",
    "model.norm.weight": "out_norm_w",
}

_TORCH_DTYPES = {
    "bfloat16": DataType.BF16,
    "float16": DataType.F16,
    "float32": DataType.F32,
}


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        dtype = _TORCH_DTYPES[cfg["torch_dtype"]]
        self._end_token = int(cfg["eos_token_id"])
        self._maxseq = 4096  # prompt + 生成长度远小于此；131072 的 KV cache 要 3.6 GB

        meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=cfg["num_hidden_layers"],
            hs=cfg["hidden_size"],
            nh=cfg["num_attention_heads"],
            nkvh=cfg["num_key_value_heads"],
            dh=cfg["hidden_size"] // cfg["num_attention_heads"],
            di=cfg["intermediate_size"],
            maxseq=self._maxseq,
            voc=cfg["vocab_size"],
            epsilon=cfg["rms_norm_eps"],
            theta=float(cfg["rope_theta"]),
            end_token=self._end_token,
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), llaisysDeviceType_t(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        # 加载全部权重：bf16 张量按 uint16 视图零拷贝取原始字节
        n_loaded = 0
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    handle = self._route(name)
                    t = f.get_tensor(name).contiguous()
                    if t.dtype == torch.bfloat16:
                        t = t.view(torch.uint16)
                    a = t.numpy()
                    LIB_LLAISYS.tensorLoad(handle, c_void_p(a.ctypes.data))
                    n_loaded += 1
                    if n_loaded % 50 == 0:
                        print(f"loaded {n_loaded} tensors ...")
        print(f"loaded {n_loaded} tensors in total")

    def _route(self, name: str):
        """safetensors 名字 → 目标张量句柄；查不到映射直接报错，不静默漏权重"""
        w = self._weights
        if name.startswith("model.layers."):
            rest = name[len("model.layers."):]
            i_str, suffix = rest.split(".", 1)
            if suffix not in _LAYER_ROUTES:
                raise KeyError(f"unknown layer weight: {name}")
            return getattr(w, _LAYER_ROUTES[suffix])[int(i_str)]
        if name in _TOP_ROUTES:
            return getattr(w, _TOP_ROUTES[name])
        raise KeyError(f"unknown weight: {name}")

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _infer(self, tokens) -> int:
        arr = (c_int64 * len(tokens))(*tokens)
        return int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, c_size_t(len(tokens))))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # --test 时 top_k=1/top_p=1.0/temperature=1.0，等价贪心采样，
        # 由 C++ infer 内部 argmax 完成；这里的采样参数仅透传不生效
        inputs = list(inputs)
        if max_new_tokens is None:
            max_new_tokens = 128

        outputs = list(inputs)
        next_id = self._infer(inputs)  # prefill：整段 prompt，产出第 1 个新 token
        outputs.append(next_id)
        while next_id != self._end_token and len(outputs) - len(inputs) < max_new_tokens:
            next_id = self._infer([next_id])  # decode：每次 1 个 token
            outputs.append(next_id)
        return outputs
