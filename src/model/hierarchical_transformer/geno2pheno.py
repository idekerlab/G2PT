import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.hierarchical_transformer import PositionWiseFeedForward, MultiHeadedAttention, MultiheadDiffAttn
from src.model.utils import poincare_log_map_zero



class Genotype2Phenotype(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2, transform=True,
                 activation='softmax', diff_transform=True, poincare=False):
        super().__init__()
        self.attn_heads = attn_heads
        self.poincare = poincare
        if diff_transform:
            self.attention = MultiheadDiffAttn(h=attn_heads, d_model=hidden, depth=1)
        else:
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout, activation=activation,
                                                  transform=transform, poincare=False)

        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.inner_norm = inner_norm
        self.outer_norm = outer_norm
        self.activation = nn.GELU()

    def forward(self, q, k, v, mask=None):
        result = self.attention.forward(q, k, v)
        result = result.squeeze(1)
        result = self.inner_norm(result)
        result = self.feed_forward(result)
        result = self.outer_norm(result)
        result = self.dropout(result)
        return result

    def get_attention(self, q, k, v):
        return self.attention.get_attention(q, k, v, mask=None)

    def get_score(self, q, k, v):
        return self.attention.get_score(q, k, v, mask=None)
