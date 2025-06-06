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


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer
    W_hat = W + (alpha / r) * B @ A
    Freezes the original weight W and learns A, B.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_seq = x.dim() == 3
        if has_seq:
            B, L, _ = x.shape
            x_flat = x.reshape(B * L, -1)
        else:
            x_flat = x
        lora_out = self.dropout(x_flat) @ self.A.t()
        lora_out = lora_out @ self.B.t() * self.scaling
        if has_seq:
            lora_out = lora_out.view(B, L, -1)
        return lora_out

class RoBERTaEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("Embedding size: ", config.vocab_size)
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
            lora=False,
            softmax=True,
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
        self.lora = lora
        if lora:
            print("Lora set True, multi-head attention level")
            self.q_lora = LoRALinear(hidden_size, hidden_size, dropout=dropout)
            self.v_lora = LoRALinear(hidden_size, hidden_size, dropout=dropout)

        # Norms
        self.softmax = softmax
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

    def _dotprod_no_softmax(self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor, p_drop: float
    ):
        """
        q,k,v : (B, L, H, d)   already split into heads
        mask  : (B, H, Lq, Lk) bool or additive-bias, or None
        returns : (B, Lq, H*d)
        """
        B, Lq, H, d = q.shape
        _, Lk, _, _ = k.shape
        q = q.permute(0, 2, 1, 3)  # (B, H, Lq, d)
        k = k.permute(0, 2, 1, 3)  # (B, H, Lk, d)
        v = v.permute(0, 2, 1, 3)  # (B, H, Lk, d)

        # 1. Scaled Dot-Product Scores
        # (B, H, Lq, d) @ (B, H, d, Lk) -> (B, H, Lq, Lk)
        scale_factor = self.scale#1.0 / math.sqrt(d)
        # Using torch.rsqrt can sometimes be slightly more optimized if d is large
        # scale_factor = torch.rsqrt(torch.tensor(d, dtype=q.dtype, device=q.device))
        scores = 1 / torch.matmul(q, k.transpose(-2, -1)) # * scale_factor

        # 2. Apply Mask (if provided) directly to the scores
        if mask is not None:
            # Ensure mask shape is compatible, potentially requires broadcasting
            # Example: mask shape (B, 1, Lq, Lk) could broadcast over H heads
            if mask.dtype == torch.bool:
                # If True means MASK OUT, set score to -infinity
                # Use masked_fill_ for in-place operation if scores tensor allows
                scores = scores.masked_fill(mask, float('-inf'))
                # If True means KEEP (less common mask convention)
                # scores = scores.masked_fill(~mask, float('-inf'))
            else:
                # Additive mask (assumes 0 where not masked, large negative where masked)
                # Ensure mask is on the same device and dtype or handle conversion
                scores = scores + mask  # Broadcasting might apply here

        # 3. Apply the (potentially masked) scores directly to Values
        #    NO SOFTMAX applied here.
        # (B, H, Lq, Lk) @ (B, H, Lk, d) -> (B, H, Lq, d)
        output = torch.matmul(scores, v)

        # Handle potential -inf * 0 = NaN cases resulting from masking if needed
        # If masked scores are -inf, and corresponding v is 0, matmul result can be NaN.
        # If this is undesirable, you might zero out NaNs, though often NaN propagation
        # is acceptable or indicates an issue elsewhere.
        # output = torch.nan_to_num(output, nan=0.0) # Optional: Replace NaN with 0

        # 4. Combine Heads: Reshape/Permute back to (B, Lq, H*d)
        # (B, H, Lq, d) -> (B, Lq, H, d)
        output = output.permute(0, 2, 1, 3)
        # Ensure contiguous memory layout after permute for reliable view/reshape
        output = output.contiguous()
        # (B, Lq, H, d) -> (B, Lq, H*d)
        output = output.view(B, Lq, H * d)
        # Alternatively: output = output.reshape(B, Lq, -1) # -1 infers H*d

        return output
    # ---------------------------------------------------------------------
    # unified wrapper  (replaces old _flash_xattn)
    # ---------------------------------------------------------------------
    def _xattn(self, q, k, v, mask):
        B = q.size(0)
        q = self.q_proj(q).view(B, -1, self.h, self.d_h)#.transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.h, self.d_h)#.transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.h, self.d_h)#.transpose(1, 2)
        if self.lora:
            q = q + self.q_lora(q)
            v = v + self.v_lora(v)
        if self.softmax:
            out = self._xops_softmax_attn(q, k, v, mask)
        else:
            out = self._dotprod_no_softmax(q, k, v, mask, self.drop.p if self.training else 0.0)

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
    def __init__(self, config, temperature=1, lora=False, softmax=True):
        super().__init__()
        if lora:
            print("Lora set true, in transformer level")
        self.self_attention = MultiHeadSelfAttention(config.hidden_size, config.num_attention_heads,
                                                     config.intermediate_size, config.dropout, temperature=temperature,
                                                     lora=lora, softmax=softmax)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.feed_forward = SwiGLUFFN(config.hidden_size, config.intermediate_size, config.dropout)
        self.layer_norm_out = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, query, key, value, residual=True):
        # Self-attention sub-layer
        attention_output = self.self_attention(query, key, value)
        x = self.layer_norm_1(query + attention_output)

        # Feed-forward sub-layer
        feed_forward_output = self.feed_forward(x)
        if residual:
            x = self.layer_norm_out(query + feed_forward_output)
        else:
            x = feed_forward_output
        return x

    def set_temperature(self, temperature):
        self.self_attention.scale = self.self_attention.scale * temperature


class RoBERTa(nn.Module):
    def __init__(self, config, num_classes, temperature=True, lora=False):
        super().__init__()
        if lora:
            print("Lora set true, in roberta level")
        self.embeddings = RoBERTaEmbedding(config)
        if temperature:
            self.transformer_layers = nn.ModuleList(
                [TransformerLayer(config, 1/np.exp2(i), lora=lora) for i, _ in enumerate(range(config.num_hidden_layers))]
            )
        else:
            self.transformer_layers = nn.ModuleList(
                [TransformerLayer(config, 1, lora=lora) for i, _ in enumerate(range(config.num_hidden_layers))]
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


class GumbelMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, temp=1.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.temp = temp

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z, mask=None):
        B, N, C = x.size()  # Batch, SeqLen, EmbeddingDim
        B, N_k, C = y.size()
        # Linear projections and split heads
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(y).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # Gumbel-Softmax instead of softmax
        logits_flat = logits.reshape(-1, N_k)  # (B*H*N, N)
        attn_flat = F.gumbel_softmax(logits_flat, tau=self.temp, hard=False, dim=-1)
        attn = attn_flat.view(B, self.num_heads, N, N_k)
        attn = self.dropout(attn)

        # Attention output
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)

class TransformerBlockGumbel(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, temp=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GumbelMultiHeadAttention(embed_dim, num_heads, dropout, temp)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y, z, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(y), self.norm1(z), mask)
        x = x + self.mlp(self.norm2(x))
        return x