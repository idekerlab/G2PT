import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.hierarchical_transformer import MultiHeadedAttention


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class HierarchicalTransformerUpdate(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, norm, dropout=0.2, norm_channel_first=False, n_type=1, transform=True, activation='softmax'):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param norm: normalization layer
        :param dropout: dropout rate
        """

        super(HierarchicalTransformerUpdate, self).__init__()
        self.attn_heads = attn_heads
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout, transform=transform, n_type=n_type, activation=activation)
        #self.layer_norm = nn.LayerNorm(hidden)
        self.norm = norm
        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_channel_first = norm_channel_first
        self.n_type = n_type

    def forward(self, q, k, v, mask=None, dropout=True):
        result = self.attention.forward(q, k, v, mask=mask, dropout=dropout)
        result = q + result
        #result_layer_norm = self.layer_norm(result)
        if self.norm_channel_first:
            result = result.transpose(-1, -2)
            result = self.norm(result)
            result = result.transpose(-1, -2)
            #result = (result + result_layer_norm)/2
        else:
            result = self.norm(result)
            #result = (result + result_layer_norm)/2
        result = self.feed_forward(result)
        #result = self.norm(result)
        return self.dropout(result)


class HierarchicalTransformer(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2, conv_type='system',
                 norm_channel_first=False, transform=True, n_type=1, activation='softmax'):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param norm: normalization layer
        :param dropout: dropout rate
        """

        super(HierarchicalTransformer, self).__init__()
        self.hierarchical_transformer_update = HierarchicalTransformerUpdate(hidden, attn_heads, feed_forward_hidden, inner_norm,
                                                                     dropout, norm_channel_first=norm_channel_first, transform=transform
                                                                             , n_type=n_type, activation=activation)
        self.norm = outer_norm
        self.conv_type = conv_type
        self.dropout = nn.Dropout(dropout)
        self.norm_channel_first = norm_channel_first
        self.n_type = n_type


    def forward(self, q, k, mask, dropout=True):
        batch_size = q.size(0)
        if self.conv_type=='system':
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1,)
        result = self.hierarchical_transformer_update(q, k, k, mask, dropout=dropout)

        #result_layer_norm = self.layer_norm(result)
        if self.norm is not None:
            if self.norm_channel_first:
                result = result.transpose(-1, -2)
                result = self.norm(result)
                result = result.transpose(-1, -2)
                #result = (result + result_layer_norm)/2
            else:
                result = self.norm(result)
                #result = (result + result_layer_norm)/2
        #updated_value = updated_value.permute(0, 2, 1)
        if self.n_type > 1:
            mask = sum([m[1] for m in mask])
        node_mask = torch.sum(mask, dim=-1) == 0
        node_mask = node_mask.unsqueeze(-1).expand(-1, -1, q.size(-1))
        result = result.masked_fill(node_mask, 0)
        return result

    def get_attention(self, q, k, v, norm=True):
        return self.hierarchical_transformer_update.attention.get_attention(q, k, v, mask=None)

    def get_score(self, q, k, v, norm=True):
        return self.hierarchical_transformer_update.attention.get_score(q, k, v, mask=None)
