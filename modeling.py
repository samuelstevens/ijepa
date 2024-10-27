"""
Defines all the models in good ol' stateful PyTorch modules.
"""

import beartype
import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


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
        grid_wp, grid_hp = torch.meshgrid(grid_wp, grid_hp, indexing="ij")

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
    """
    Module to patchify images and project from pixel space to embedding space.
    """

    def __init__(self, ch: int, output_d: int, patch_size: int):
        """
        Args:
            ch: Number of channels in input images (almost always 3).
            output_d: Output dimension.
            patch_size: Length of one size of a patch.
        """
        super().__init__()
        self.patch_size = patch_size
        self.linear = torch.nn.Linear(patch_size * patch_size * ch, output_d)

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
    """
    A pre-norm transformer layer, combining a self-attention layer without any masking and a single-layer MLP.
    """

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
        # all_mask is a mask that contains all the patches.
        width_p, height_p = size_p
        self.register_buffer("all_mask", torch.arange(width_p * height_p))

    def forward(
        self,
        x: Float[Tensor, "batch 3 width height"],
        mask: None | Int[Tensor, " patches"] = None,
    ) -> Float[Tensor, "batch patches d"]:
        if mask is None:
            mask = self.all_mask
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

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        d_vit: int,
        size_p: tuple[int, int],
    ):
        """
        Args:
            d: Transformer residual dimension.
            n_heads: Number of transformer attention heads.
            n_layers: Number of transformer layers.
            p_dropout: Probability of dropout in transformer layers.
            d_vit: ViT dimension (input to this model).
            size_p: Size of inputs in patches.
        """
        super().__init__()

        self.proj_in = torch.nn.Linear(d_vit, d)
        self.pos_embd = PositionalEmbedding(d, size_p)
        # TODO: initialize the mask_token with a truncated normal.
        self.mask_token = torch.nn.Parameter(torch.zeros(d))

        # Use 4/3x expansion instead of 4x.
        # TODO: check on this.
        d_mlp = int(d * 4 / 3)

        self.transformer = Transformer(d, d_mlp, n_heads, n_layers, p_dropout)
        self.proj_out = torch.nn.Linear(d, d_vit)

    def forward(
        self,
        x: Float[Tensor, "batch n_ctx_patches d_vit"],
        tgt_masks: Int[Tensor, " n_tgt_patches"],
    ) -> Float[Tensor, "batch n_tgt_patches d_vit"]:
        """
        Given `x`, the output of the context encoder, we want to predict the patch-level target block representations. To do that, we combine a learnable mask token with a fixed position embedding for the target patches.
        """
        (n_tgt_patches,) = tgt_masks.shape
        batch_size, _, _ = x.shape

        x = self.proj_in(x)

        masked = self.mask_token[None, :] + self.pos_embd(tgt_masks)
        masked = masked.expand(batch_size, -1, -1)

        x = torch.cat((masked, x), dim=1)
        x = self.transformer(x)

        # Ignore representations for x; we only care about the tgt patches.
        x = x[:, :n_tgt_patches, :]
        x = self.proj_out(x)
        return x
