""" """

import beartype

from jaxtyping import Float, Int, jaxtyped
import torch.nn.functional as F
import torch
from torch import Tensor
import einops


@beartype.beartype
class PositionalEmbedding(torch.nn.Module):
    """
    See this [notebook](https://marimo.app/l/own8um) for an interactive notebook to understand how 2D sin-cos positional embeddings work.
    """

    def __init__(self, d: int, size_p: tuple[int, int], temperature: float = 10000.0):
        """
        Args:
            d: Embedding dimension
            size_p: Size of the inputs in patches.
            temperature: How to scale sin/cos period.
        """
        super().__init__()
        err_msg = "Embed dim must be divisible by 4 for 2D sin-cos position embedding"
        assert d % 4 == 0, err_msg
        wp, hp = size_p
        grid_wp = torch.arange(wp, dtype=torch.float32)
        grid_hp = torch.arange(hp, dtype=torch.float32)
        grid_wp, grid_hp = torch.meshgrid(grid_wp, grid_hp)

        pos_d = d // 4
        omega = torch.arange(pos_d, dtype=torch.float32) / pos_d
        omega = 1.0 / (temperature**omega)

        out_w = torch.einsum("m,d->md", grid_wp.flatten(), omega)
        out_h = torch.einsum("m,d->md", grid_hp.flatten(), omega)

        pos_embd = torch.cat(
            [
                torch.sin(out_w),
                torch.cos(out_w),
                torch.sin(out_h),
                torch.cos(out_h),
            ],
            axis=1,
        )
        self.register_buffer("pos_embd", pos_embd)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, pos: Int[Tensor, " patches"]) -> Float[Tensor, "patches d"]:
        return self.pos_embd[pos]


@beartype.beartype
class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_ch: int, output_d: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.linear = torch.nn.Linear(patch_size * patch_size * in_ch, output_d)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, x: Float[Tensor, "batch channels width_i height_i"]
    ) -> Float[Tensor, "batch patches d"]:
        x = einops.rearrange(
            x,
            "b c (w pw) (h ph) -> b (w h) (c pw ph)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.linear(x)
        return x


@beartype.beartype
class AttentionBlock(torch.nn.Module):
    def __init__(self, d: int, d_mlp: int, n_heads: int, p_dropout: float):
        """
        Args:
            d: Transformer residual dimension.
            d_mlp: Hidden layer dimension of the MLPs.
            n_heads: Number of attention heads.
            p_dropout: Probability of dropout in attention and MLP layers.
        """
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d)
        self.ln2 = torch.nn.LayerNorm(d)
        self.attn = torch.nn.MultiheadAttention(
            d, n_heads, dropout=p_dropout, bias=False, batch_first=True
        )

        self.linear1 = torch.nn.Linear(d, d_mlp)
        self.dropout = torch.nn.Dropout(p_dropout)
        self.linear2 = torch.nn.Linear(d_mlp, d)

    def forward(
        self, x: Float[Tensor, "batch patches d"]
    ) -> Float[Tensor, "batch patches d"]:
        x_ = self.ln1(x)
        x_, _ = self.attn(x_, x_, x_, need_weights=False)
        x = x + x_

        x_ = self.ln2(x)
        x_ = self.linear1(x_)
        x_ = F.gelu(x_)
        x_ = self.dropout(x_)
        x_ = self.linear2(x_)

        x = x + x_
        return x


@beartype.beartype
class Transformer(torch.nn.Module):
    """
    Generic bidirectional transformer.
    """

    def __init__(
        self, d: int, d_mlp: int, n_heads: int, n_layers: int, p_dropout: float
    ):
        """
        Args:
            d: Transformer residual dimension.
            d_mlp: Hidden layer dimension of the MLPs.
            n_heads: Number of attention heads.
            n_layers: Number of layers.
            p_dropout: Probability of dropout in attention and MLP layers.
        """
        super().__init__()

        self.attn_blocks = torch.nn.ModuleList([
            AttentionBlock(d, d_mlp, n_heads, p_dropout) for _ in range(n_layers)
        ])

    def forward(self, x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
        """Feed `x` through the sequence of attention blocks."""
        for block in self.attn_blocks:
            x = block(x)
        return x


@beartype.beartype
class VisionTransformer(torch.nn.Module):
    """
    Vision transformer with patch embedding to project from pixel space to residual space and fixed 2D sin/cos position embedding.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        patch_size: int,
        size_p: tuple[int, int],
    ):
        """
        Args:
            d: Transformer residual dimension.
            n_heads: Number of transformer attention heads.
            n_layers: Number of transformer layers.
            p_dropout: Probability of dropout in transformer layers.
            patch_size: Length of one patch side in pixels.
            size_p: Size of inputs in patches.
        """
        super().__init__()

        self.patch_embd = PatchEmbedding(3, d, patch_size)
        self.pos_embd = PositionalEmbedding(d, size_p)

        # Use 4/3x expansion instead of 4x.
        # TODO: check on this.
        d_mlp = int(d * 4 / 3)

        self.transformer = Transformer(d, d_mlp, n_heads, n_layers, p_dropout)

    def forward(
        self, x: Float[Tensor, "batch 3 width height"], mask: Int[Tensor, " patches"]
    ) -> Float[Tensor, "batch patches d"]:
        x = self.patch_embd(x)
        x = x[:, mask, :]
        x = x + self.pos_embd(mask)

        x = self.transformer(x)

        return x


@beartype.beartype
class PredictorTransformer(torch.nn.Module):
    """
    Narrow predictor transformer for predicting the target representations from the context representations and the positional embeddings.
    """

    def __init__(self):
        super().__init__()
