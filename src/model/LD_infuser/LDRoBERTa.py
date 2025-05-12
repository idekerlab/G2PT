import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import xformers.ops as xops

class RoBERTaConfig:
    def __init__(self,
                 vocab_size=50265,
                 max_position_embeddings=514,
                 hidden_size=768,
                 num_attention_heads=12,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout


class BlockAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=False))
    def forward(self, x):              # x: [B, L_blk, H_in]
        return self.proj(x)


class RoBERTaEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,  # e.g. 512 or 1024
            config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        token_embeds = self.token_embeddings(input_ids)
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)

        # Normalize embeddings: force length to 1 and mean to 0
        embeddings = self.layer_norm(embeddings)

        return embeddings


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float):  # d_ff keeps API stable
        super().__init__()
        # We still accept d_ff but compute expansion ratio from it.
        expansion = (d_ff // d_model)
        inner = expansion * d_model
        self.w12 = nn.Linear(d_model, inner * 2, bias=False)  # gate + value
        self.proj = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        u, v = self.w12(x).chunk(2, dim=-1)
        return self.proj(self.drop(F.silu(u) * v))



class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            intermediate_size: int,
            dropout: float = 0.2,
            temperature: float = 1,
            **kwargs
    ) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden must divide heads"
        self.h = num_attention_heads
        self.d_h = hidden_size // num_attention_heads


        # Independent projections for true cross‑attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = math.sqrt(hidden_size) * temperature

        # Norms
        self.norm_attn = nn.LayerNorm(hidden_size, eps=1e-5)  # pre‑supplied
        #self.norm_ffn = nn.LayerNorm(hidden_size, eps=1e-5)
        #self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)
        self.drop = nn.Dropout(dropout)



    def _xops_softmax_attn(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            mask
    ):
        B, Lq, H, dh = q.shape
        # print(q.size(), k.size(), mask.size())
        # qh = q.reshape(B * H, Lq, dh)
        # kh = k.reshape(B * H, k.size(2), dh)
        # vh = v.reshape(B * H, v.size(2), dh)

        bias = None
        if mask is not None:  # (B,1|H,Lq,Lk)
            mask = mask.unsqueeze(1).repeat(1, 4, 1, 1)  # mask.expand(B, H, -1, -1).reshape(B * H, Lq, -1)
            # bias = xops.fmha.attn_bias.PreconvertedBias(mask, True)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=mask,
            p=self.drop.p if self.training else 0.0,
            scale = self.scale,
        )
        # print(out.shape)# (B, lq, H,dh)
        return out.reshape(B, Lq, H, dh) \
            .reshape(B, Lq, H * dh)

    # ---------------------------------------------------------------------
    # no-softmax path  (keeps behaviour of your original helper)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # unified wrapper  (replaces old _flash_xattn)
    # ---------------------------------------------------------------------
    def _xattn(self, q, k, v, mask):
        B = q.size(0)
        q = self.q_proj(q).view(B, -1, self.h, self.d_h)  # .transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.h, self.d_h)  # .transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.h, self.d_h)  # .transpose(1, 2)
        out = self._xops_softmax_attn(q, k, v, mask)
        return self.o_proj(out)

    # ---------------------------------------------------------------------
    # forward is unchanged except the private call name
    # ---------------------------------------------------------------------
    def forward(self, q, k, v, mask=None):
        q_norm = self.norm_attn(q)
        x = q + self.drop(self._xattn(q_norm, k, v, mask))
        #y = self.norm_ffn(x)
        #x = x + self.drop(self.ffn(y))
        #x = self.norm_attn(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, config, temperature=1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(config.hidden_size, config.num_attention_heads,
                                                     config.intermediate_size, config.dropout, temperature=temperature)
        #self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.feed_forward = SwiGLUFFN(config.hidden_size, config.intermediate_size, config.dropout)
        self.layer_norm_out = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, query, key, value):
        # Self-attention sub-layer
        attention_output = self.self_attention(query, key, value)
        #x = self.layer_norm_1(x + attention_output)

        # Feed-forward sub-layer
        feed_forward_output = self.feed_forward(attention_output)
        x = self.layer_norm_out(query + feed_forward_output)
        return x

class RoBERTa(nn.Module):
    def __init__(self, config, num_classes, temperature=True):
        super().__init__()
        self.embeddings = RoBERTaEmbedding(config)
        if temperature:
            self.transformer_layers = nn.ModuleList(
                [TransformerLayer(config, 1/np.exp2(i)) for i, _ in enumerate(range(config.num_hidden_layers))]
            )
        else:
            self.transformer_layers = nn.ModuleList(
                [TransformerLayer(config, 1) for i, _ in enumerate(range(config.num_hidden_layers))]
            )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def predict(self, input_ids):
        x = self.forward(input_ids)
        output = self.pooler(x)
        output = self.pooler_activation(output)

        logits = self.classifier(output)
        return logits

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.transformer_layers:
            x = layer(x, x, x)
        return x
