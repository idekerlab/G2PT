import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.hierarchical_transformer import PositionWiseFeedForward, MultiheadDiffAttn, HierarchicalTransformer



class System2Phenotype(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2,
                 transform=True, activation='softmax', poincare=False, use_hierarchical_transformer=False):
        super().__init__()
        self.attn_heads = attn_heads
        self.use_hierarchical_transformer = use_hierarchical_transformer
        if use_hierarchical_transformer:
            self.attention = HierarchicalTransformer(hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout,
                                                     conv_type='system', norm_channel_first=False, transform=transform,
                                                     n_type=1, activation=activation, poincare=poincare)
        else:
            self.attention = MultiheadDiffAttn(h=attn_heads, d_model=hidden, depth=1)
        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.inner_norm = inner_norm
        self.outer_norm = outer_norm
        self.activation = nn.GELU()

    def forward(self, q, k, v, mask=None):
        if self.use_hierarchical_transformer:
            result = self.attention.forward(q, k, mask)
        else:
            result = self.attention.forward(q, k, v)
        result = result.squeeze(1)
        result = self.inner_norm(result)
        result = self.feed_forward(result)
        result = self.outer_norm(result)
        result = self.dropout(result)
        return result

    def get_attention(self, q, k, v):
        if self.use_hierarchical_transformer:
            return self.attention.get_attention(q, k, None)
        else:
            return self.attention.get_attention(q, k, v)

    def get_score(self, q, k, v):
        if self.use_hierarchical_transformer:
            return self.attention.get_score(q, k, None)
        else:
            return self.attention.get_score(q, k, v)