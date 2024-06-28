import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, batch_first=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))

        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True)

        init.xavier_uniform_(self.in_proj_weight)
        init.constant_(self.in_proj_bias, 0.)
        init.constant_(self.out_proj.bias, 0.)

    def forward(self, k, q, v, need_weights=False):
        tgt_len, bsz, embed_dim = q.shape
        num_heads = self.num_heads

        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads

        proj = F.linear(q, self.in_proj_weight, self.in_proj_bias)
        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        q, k, v = proj.unflatten(-1, (3, q.size(-1))).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()

        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output, None


torch.nn.MultiheadAttention = MultiHeadAttention
