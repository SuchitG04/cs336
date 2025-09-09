import math

import torch
from einops import einsum, rearrange, repeat
from torch import nn


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()

        std = math.sqrt(2 / (in_features + out_features))
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype), 0, std, a=-3 * std, b=3 * std
            ),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 0, 1, a=-3, b=3
            ),
            requires_grad=True,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # (..., d_model)*(..., 1)->(..., d_model)
        return (norm * self.gain).to(in_dtype)


class SwiGLU(nn.Module):
    """
    FFN(x) = SwiGLU(x, w_1, w_2, w_3) = w_2*(SiLU(w_1*x) ⊙ w_3*x)
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x) -> torch.Tensor:
        w1_x = self.w1(x)  # (..., d_model) -> (..., d_ff)
        silu_x = w1_x * torch.sigmoid(w1_x)  # SiLU or Swish
        return self.w2(silu_x * self.w3(x))  # w2(hadamard product or gating)


class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ):
        super().__init__()
        inv_freq = theta ** -(torch.arange(0, d_k, 2, device=device).float() / d_k)  # (d_k/2)
        positional_thetas = einsum(torch.arange(max_seq_len, device=device).float(), inv_freq, 'msl, dk_2 -> msl dk_2')
        # positional_thetas = torch.repeat_interleave(positional_thetas, 2, dim=-1)
        # or using einops ...
        positional_thetas = rearrange(
            repeat(positional_thetas, 'msl dk_2 -> msl dk_2 2'),
            'msl dk_2 two -> msl (dk_2 two)'
        )
        self.register_buffer("cos_cache", positional_thetas.cos())
        self.register_buffer("sin_cache", positional_thetas.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_c = torch.empty_like(x)
        x_c[..., 1::2] = x[..., ::2]
        x_c[..., ::2] = -x[..., 1::2]
        return (x * self.cos_cache[token_positions, :].to(x.dtype)) + \
            (x_c * self.sin_cache[token_positions, :].to(x_c.dtype))


def softmax(
        x: torch.Tensor,
        dim: int
) -> torch.Tensor:
    x -= x.max(dim=dim, keepdim=True).values
    return x.exp() / x.exp().sum(dim=dim, keepdim=True, dtype=x.dtype)


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
) -> torch.Tensor:
    qk = einsum(q, k, '... q d_k, ... k d_k -> ... q k')
    qk_normed = qk / torch.sqrt(torch.tensor(k.shape[-1]))
    if mask is not None:
        # set to -inf where mask element is False
        qk_normed = torch.where(
            condition=mask,
            input=qk_normed,
            other=torch.fill(torch.empty_like(qk_normed), -torch.inf),
        )
    attn_out = einsum(softmax(qk_normed, -1), v, '... q k, ... k d_v -> ... q d_v')
    return attn_out


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            rope: RoPE | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        hdk = d_model
        self.wq = Linear(d_model, hdk)
        self.wk = Linear(d_model, hdk)
        self.wv = Linear(d_model, hdk)
        self.wo = Linear(hdk, d_model)
        if rope:
            self.rope = rope

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = rearrange(q, '... T (nh d_k) -> ... nh T d_k', nh=self.num_heads)
        k = rearrange(k, '... T (nh d_k) -> ... nh T d_k', nh=self.num_heads)
        v = rearrange(v, '... T (nh d_k) -> ... nh T d_k', nh=self.num_heads)
        if hasattr(self, "rope"):
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        attn_mask = torch.tril(torch.fill(torch.empty(x.shape[-2], x.shape[-2]), True).to(torch.bool))
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask)
        attn_out = rearrange(attn_out, '... nh T d_k -> ... T (nh d_k)')
        return self.wo(attn_out)


class Block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RoPE,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope
        )
        self.ln_mha = RMSNorm(d_model)
        self.ln_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        token_positions = torch.arange(x.shape[-2])
        token_positions = token_positions.expand((x.shape[:-1]))
        x = x + self.mha(self.ln_mha(x), token_positions)
        return x + self.ffn(self.ln_ffn(x))


class TransformerLM(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            vocab_size: int,
            num_layers: int,
            rope: RoPE,
    ):
        super().__init__()
        self.tok_emb = Embedding(vocab_size, d_model)
        self.tfmr_blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=rope,
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices):
        x = self.tok_emb(in_indices)
        for block in self.tfmr_blocks:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

__all__ = [
    "Linear",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "RoPE",
    "softmax",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "Block",
    "TransformerLM",
]
