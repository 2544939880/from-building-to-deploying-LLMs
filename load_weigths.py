import architecture
import torch
from transformers import GPT2Model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

def load_weigths_gpt2_hf(gpt: architecture.GPTModel, CHOOSE_MODEL = "gpt2-small (124M)"):
    # allowed model names
    model_names = {
        "gpt2-small (124M)": "openai-community/gpt2",
        "gpt2-medium (355M)": "openai-community/gpt2-medium",
        "gpt2-large (774M)": "openai-community/gpt2-large",
        "gpt2-xl (1558M)": "openai-community/gpt2-xl"
    }
    BASE_CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "dropout": 0.0,         # Dropout rate
        "qkv_bias": True        # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    gpt_hf = GPT2Model.from_pretrained(
        model_names[CHOOSE_MODEL], cache_dir="checkpoints")
    
    d = gpt_hf.state_dict()
    # l = gpt.state_dict()
    # d_keys = [iter for iter in d.keys()]
    # l_keys = [iter for iter in l.keys()]
    # print("*"*40)
    # for i in range(15):
    #     print(d_keys[i], d[d_keys[i]].shape)
    # print("*"*40)
    # for i in range(20):
    #     print(l_keys[i-20], l[l_keys[i-20]].shape)

    gpt.token_emb.weight = assign_check(gpt.token_emb.weight, gpt_hf.wte.weight)
    gpt.position_emb.weight = assign_check(gpt.position_emb.weight, gpt_hf.wpe.weight)
    
    for b in range(BASE_CONFIG["n_layers"]):
        W_q, W_k, W_v = torch.chunk(d[f"h.{b}.attn.c_attn.weight"], 3, dim=-1)
        gpt.trf_blocks[b].msa.W_query.weight = assign_check(gpt.trf_blocks[b].msa.W_query.weight, W_q.T)
        gpt.trf_blocks[b].msa.W_key.weight = assign_check(gpt.trf_blocks[b].msa.W_key.weight, W_k.T)
        gpt.trf_blocks[b].msa.W_value.weight = assign_check(gpt.trf_blocks[b].msa.W_value.weight, W_v.T)

        b_q, b_k, b_v = torch.chunk(d[f"h.{b}.attn.c_attn.bias"], 3, dim=-1)
        gpt.trf_blocks[b].msa.W_query.bias = assign_check(gpt.trf_blocks[b].msa.W_query.bias, b_q)
        gpt.trf_blocks[b].msa.W_key.bias = assign_check(gpt.trf_blocks[b].msa.W_key.bias, b_k)
        gpt.trf_blocks[b].msa.W_value.bias = assign_check(gpt.trf_blocks[b].msa.W_value.bias, b_v)

        gpt.trf_blocks[b].msa.out_proj.weight = assign_check(gpt.trf_blocks[b].msa.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].msa.out_proj.bias = assign_check(gpt.trf_blocks[b].msa.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(gpt.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(gpt.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(gpt.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(gpt.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])
    
        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign_check(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign_check(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

    gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d[f"ln_f.weight"])
    gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d[f"ln_f.bias"]) 
    gpt.out_proj.weight = assign_check(gpt.out_proj.weight, d["wte.weight"])

    return gpt


'''
BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "dropout": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

gpt = architecture.GPTModel(BASE_CONFIG)

load_weigths(gpt, gpt_hf)
'''