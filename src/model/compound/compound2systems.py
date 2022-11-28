import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.tree_conv import PositionWiseFeedForward, MultiHeadedAttention



class CompoundToSystems(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2):
        super().__init__()
        self.attn_heads = attn_heads
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout, activation='softmax')
        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.inner_norm = inner_norm
        self.outer_norm = outer_norm
        self.activation = nn.GELU()
        #self.norm_channel_first = norm_channel_first

    def forward(self, q, k, v, mask=None):
        #q = self.dropout(q / torch.norm(q, p=2, dim=-1, keepdim=True))
        #k = self.dropout(k / torch.norm(k, p=2, dim=-1, keepdim=True))
        #v = self.dropout(v / torch.norm(v, p=2, dim=-1, keepdim=True))
        #mask = torch.tensor(torch.sum(k, dim=-1)!=0, dtype=torch.float32).unsqueeze(1).unsqueeze(1).clone().detach()
        #if self.norm is not None:
        #    q = self.norm(q)
        #    k = self.norm(k)
        #   v = self.norm(v)
        #mask = torch.ones_like(torch.sum(q, dim=-1))
        result = self.attention.forward(q, k, v, mask=mask)
        result = result.squeeze(1)
        result = self.inner_norm(result)
        result = self.feed_forward(result)
        result = self.outer_norm(result)

        #result = self.activation(result)
        result = self.dropout(result)
        return result

    def get_attention(self, q, k, v):
        #q = self.dropout(q / torch.norm(q, p=2, dim=-1, keepdim=True))
        #k = self.dropout(k / torch.norm(k, p=2, dim=-1, keepdim=True))
        #v = self.dropout(v / torch.norm(v, p=2, dim=-1, keepdim=True))
        return self.attention.get_attention(q, k, v, mask=None)