import copy
import dataclasses
import logging
import math

import beartype
import einops
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import tyro
import webdataset as wds
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import modeling

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("main")


IMAGENET_CHANNEL_MEAN = (0.4632, 0.4800, 0.3762)
IMAGENET_CHANNEL_STD = (0.2375, 0.2291, 0.2474)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configuration for a training run."""

    batch_size: int = 2048
    """Training batch size."""
    n_epochs: int = 40
    """Number of training epochs."""

    # Data
    data_url: str = "/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar"
    """Path to webdataset shards."""
    n_workers: int = 4
    """Number of workers to load data with."""
    resize_size: int = 256
    """How big to resize images."""
    crop_size: int = 224
    """After resize, how big an image to crop."""

    # I-JEPA Objective
    tgt_mask_scale: tuple[float, float] = (0.15, 0.2)
    """Target block mask size range."""
    tgt_mask_aspect_ratio: tuple[float, float] = (0.75, 1.5)
    """Target block mask aspect ratio range."""
    n_tgt_blocks: int = 4
    """Number of target blocks."""
    ctx_mask_scale: tuple[float, float] = (0.85, 1.0)
    """Context block mask size range."""
    ctx_mask_aspect_ratio: tuple[float, float] = (1.0, 1.0)
    """Context block mask aspect ratio range."""
    min_ctx_patches: int = 10
    """Minimum number of context patches."""

    # Modeling
    d_vit: int = 768
    """ViT's residual dimension."""
    n_vit_heads: int = 12
    """Number of attention heads for ViT."""
    n_vit_layers: int = 12
    """Number of ViT layers."""
    p_vit_dropout: float = 0.1
    """Probability of dropout in ViT."""
    patch_size: int = 16
    """Patch size for the ViT."""

    # Misc
    device: str = "cuda"


@beartype.beartype
class MaskCollator:
    """
    MaskCollator chooses masks for a given batch.

    I-JEPA makes use of several masks.

    First, the context mask is the set of patches that the context encoder embeds. These masks are referred to with the `ctx_` prefix.

    Second, the target masks is the set of patch sets that the narrow predictor network tries to predict. These masks and patches are referred to with the `tgt_` prefix.

    The context mask cannot overlap with the target mask, but the context mask must have a minimum number of patches (`Args.min_ctx_patches`).

    TODO: make it not int64 because it's very inefficient for GPUs.
    """

    def __init__(self, args: Args):
        self.tgt_mask_scale = args.tgt_mask_scale
        self.tgt_mask_aspect_ratio = args.tgt_mask_aspect_ratio
        self.n_tgt_blocks = args.n_tgt_blocks

        self.ctx_mask_scale = args.ctx_mask_scale
        self.ctx_mask_aspect_ratio = args.ctx_mask_aspect_ratio
        self.min_ctx_patches = args.min_ctx_patches

        assert args.crop_size % args.patch_size == 0
        self.size = (
            args.crop_size // args.patch_size,
            args.crop_size // args.patch_size,
        )

    def sample_block_size(
        self,
        scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        size: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Sample a (width', height') that is within the scale range and the aspect ratio range with respect to the original size (width, height).

        Note that once we know the sampled scale and aspect ratio, then:

        w' * h' = w * h * scale

        w' / h' = aspect

        Then with algebra we can find the expresions for h' and w'.
        """
        r = torch.rand(1).item()
        min_s, max_s = scale
        mask_scale = min_s + r * (max_s - min_s)

        min_a, max_a = aspect_ratio
        mask_aspect_ratio = min_a + r * (max_a - min_a)

        w, h = size

        h_ = math.sqrt(w * h * mask_scale / mask_aspect_ratio)
        w_ = int(round(mask_aspect_ratio * h_))
        h_ = int(round(h_))

        # Can't be bigger than original size.
        h_ = min(h_, h)
        w_ = min(w_, w)

        return w_, h_

    @jaxtyped(typechecker=beartype.beartype)
    def sample_block_mask(
        self,
        block_size_p: tuple[int, int],
        size_p: tuple[int, int],
        *,
        unacceptable: list[Int[Tensor, "w_patch h_patch"]],
    ) -> Int[Tensor, "w_patch h_patch"]:
        """
        Sample a block mask. When picking the context mask, we have to ignore the target masks. If it's impossible to sample a context mask that satisfies the overlap constraints, we will allow some overlap between target masks and the context masks.

        Args:
            block_size_p: block width and height in patches.
        """

        def constrain_mask(mask: Int[Tensor, "width height"]):
            for complement in unacceptable:
                mask *= 1 - complement

        b_wp, b_hp = block_size_p
        wp, hp = size_p
        valid = False
        while not valid:
            top = torch.randint(0, hp - b_hp + 1, (1,))
            left = torch.randint(0, wp - b_wp + 1, (1,))
            mask = torch.zeros(size_p, dtype=int)
            mask[left : left + b_wp, top : top + b_hp] = 1
            if unacceptable:
                constrain_mask(mask)
            valid = mask.sum().item() >= self.min_ctx_patches

            if not valid:
                print("invalid")

        return mask

    @jaxtyped(typechecker=beartype.beartype)
    def __call__(
        self, batch: list[Float[Tensor, "batch 3 w_img h_img"]]
    ) -> tuple[
        Float[Tensor, "batch 3 w_img h_img"],
        Int[Tensor, "n patches"],
        Int[Tensor, " patches"],
    ]:
        """ """
        assert len(batch) == 1, f"len(batch) == {len(batch)} != 1."
        batch = batch[0]

        tgt_masks = []

        for _ in range(self.n_tgt_blocks):
            block_size = self.sample_block_size(
                self.tgt_mask_scale, self.tgt_mask_aspect_ratio, self.size
            )
            tgt_masks.append(
                self.sample_block_mask(block_size, self.size, unacceptable=[])
            )

        block_size = self.sample_block_size(
            self.ctx_mask_scale, self.ctx_mask_aspect_ratio, self.size
        )
        ctx_mask = self.sample_block_mask(block_size, self.size, unacceptable=tgt_masks)

        ctx_mask = einops.rearrange(ctx_mask, "wp hp -> (wp hp)")
        tgt_masks = torch.stack(tgt_masks)
        tgt_masks = einops.rearrange(tgt_masks, "n wp hp -> n (wp hp)")

        return batch, tgt_masks, ctx_mask


def filter_no_caption_or_no_image(sample):
    has_caption = any("txt" in key for key in sample)
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logger.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


@beartype.beartype
def make_dataloader(args: Args) -> torch.utils.data.DataLoader:
    img_transform = transforms.Compose([
        transforms.Resize(args.resize_size, antialias=True),
        transforms.RandomResizedCrop(
            args.crop_size,
            scale=(0.08, 1.0),
            ratio=(0.75, 4.0 / 3.0),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2, 10),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD),
    ])

    # TODO: shuffle
    dataset = wds.DataPipeline(
        # at this point we have an iterator over all the shards
        wds.SimpleShardList(args.data_url),
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp"),
        wds.map_dict(image=img_transform),
        wds.to_tuple("image"),
        wds.batched(args.batch_size, partial=True),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.n_workers,
        persistent_workers=args.n_workers > 0 and args.n_epochs > 1,
        collate_fn=MaskCollator(args),
    )
    return dataloader


@jaxtyped(typechecker=beartype.beartype)
def make_all_mask(size_p: tuple[int, int]) -> Int[Tensor, " patches"]:
    width_p, height_p = size_p
    return torch.arange(width_p * height_p)


@beartype.beartype
def main(args: Args):
    size_p = (args.crop_size // args.patch_size, args.crop_size // args.patch_size)

    ##########
    # Models #
    ##########
    ctx_model = modeling.VisionTransformer(
        args.d_vit,
        args.n_vit_heads,
        args.n_vit_layers,
        args.p_vit_dropout,
        args.patch_size,
        size_p,
    )
    # tgt_model is an exponential moving average (EMA) of ctx_model.
    tgt_model = copy.deepcopy(ctx_model)
    pred_model = modeling.PredictorTransformer()

    dataloader = make_dataloader(args)

    all_mask = make_all_mask(size_p)

    for epoch in range(args.n_epochs):
        logger.info("Epoch %d.", epoch + 1)

        for b, (imgs, tgt_masks, ctx_mask) in enumerate(dataloader):
            imgs = imgs.to(args.device, non_blocking=True)
            tgt_masks = tgt_masks.to(args.device, non_blocking=True)
            ctx_mask = ctx_mask.to(args.device, non_blocking=True)

            with torch.no_grad():
                h = tgt_model(imgs, all_mask)
            z = ctx_model(imgs, ctx_mask)
            z = pred_model(z, ctx_mask, tgt_masks)

            loss = F.smooth_l1_loss(z, h)
            breakpoint()
            loss.backward()


if __name__ == "__main__":
    main(tyro.cli(Args))
