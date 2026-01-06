import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.hierarchical_transformer import MultiheadDiffAttn, SwiGLUFFN, HierarchicalTransformer



class Genotype2Phenotype(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2, attention_dropout=0.1, transform=True,
                 activation='softmax', diff_transform=True, use_hierarchical_transformer=False):
        super().__init__()
        self.attn_heads = attn_heads
        self.use_hierarchical_transformer = use_hierarchical_transformer
        if use_hierarchical_transformer:
            self.attention = HierarchicalTransformer(hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout,
                                                     attention_dropout=attention_dropout,
                                                     conv_type='system', norm_channel_first=False, transform=transform,
                                                     n_type=1, activation=activation)
        elif diff_transform:
            self.attention = MultiheadDiffAttn(h=attn_heads, d_model=hidden, depth=0, attention_dropout=attention_dropout)

        self.feed_forward = SwiGLUFFN(d_model=hidden, d_ff=feed_forward_hidden, p=dropout)
        self.dropout = nn.Dropout(dropout)
        self.inner_norm = inner_norm
        self.outer_norm = outer_norm
        self.activation = nn.GELU()

    def forward(self, q, k, v, mask=None):
        if self.use_hierarchical_transformer:
            result = self.attention.forward(q, k, mask)
        else:
            result = self.attention.forward(q, k, v)
        result = self.inner_norm(result)
        result = q + result
        #if len(result.size()) > 3:
        #result = result.squeeze(1)
        result = self.feed_forward(result)
        result = self.outer_norm(result)
        result = self.dropout(result)
        return result

    def get_attention(self, q, k, v):
        if self.use_hierarchical_transformer:
            return self.attention.get_attention(q, k, None)
        else:
            return self.attention.get_attention(q, k, v, mask=None)

    def get_score(self, q, k, v):
        if self.use_hierarchical_transformer:
            return self.attention.get_score(q, k, None)
        else:
            return self.attention.get_score(q, k, v, mask=None)
