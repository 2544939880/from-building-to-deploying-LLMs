import torch
import torch.nn as nn

from einops import rearrange

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token and position embeddings
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
    
        # Final normalization layer and output projection
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, num_tokens = x.shape
        token_embeds = self.token_emb(x)
        position_embeds = self.position_emb(torch.arange(num_tokens).to(x.device))

        # Add token and position embeddings
        # x shape: [batch_size, num_tokens, emb_dim]
        x = token_embeds + position_embeds  
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x = self.out_proj(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Gaussian Error Linear Unit activation
        out = 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3))
        ))

        return out
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class MultiHeadCausalAttention(nn.Module):
    # d_in: input token embed dim, d_out: q, k, and v space dim
    def __init__(self, d_in, d_out, sample_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_out / num_heads
    
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                             torch.triu(torch.ones(sample_length, sample_length), diagonal=1))
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        # x shape: [batch_size, num_tokens, d_in]
        batch_size, num_tokens, d_in = x.shape

        # q, k, and v shape: [batch_size, num_tokens, d_out]
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split the matrix by adding a "num_heads" dimension
        # q, k, and v shape: [batch_size, num_heads, num_tokens, head_dim]
        queries, keys, values = map(
            lambda x: rearrange(x, 'b s (h d) -> b h s d', h=self.num_heads),
            [queries, keys, values]
        )

        attn_scores = queries @ keys.transpose(2, 3)

        # Set mask size according to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores * (keys.shape[-1] ** -0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        attention = attn_weights @ values

        # attention shape: [batch_size, num_tokens, d_out]
        attention = rearrange(attention, 'b h s d -> b s (h d)')
        attention = self.out_proj(attention)

        return attention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.msa = MultiHeadCausalAttention(
            d_in=cfg["emb_dim"], 
            d_out=cfg["emb_dim"], 
            sample_length=cfg["context_length"], 
            num_heads=cfg["n_heads"], 
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.msa(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        out = shortcut + x

        return out
