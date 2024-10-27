"""
TODO:

1. Weight decay
2. Distributed data parallel
3. Checkpointing
4. Shuffle data
"""

import copy
import dataclasses
import logging
import math
import time
import typing

import beartype
import einops
import submitit
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import tyro
import webdataset as wds
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import modeling
import newt
import schedulers
import wandb

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


IMAGENET_CHANNEL_MEAN = (0.4632, 0.4800, 0.3762)
IMAGENET_CHANNEL_STD = (0.2375, 0.2291, 0.2474)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configuration for a training run."""

    batch_size: int = 128
    """Training batch size (per device)."""
    n_epochs: int = 40
    """Number of training epochs."""

    # Data
    data_url: str = "/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar"
    """Path to webdataset shards."""
    n_imgs: int = 9562377
    """Number of images in the training data."""
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

    d_pred: int = 384
    """Predictor transformer's residual dimension."""
    n_pred_heads: int = 12
    """Number of predictor transformer attention heads (kept same as ViT)."""
    n_pred_layers: int = 6
    """Number of predictor transformer layers."""
    p_pred_dropout: float = 0.1
    """Probability of dropout in predictor transformer."""

    # Optimization
    lr_init: float = 1e-4
    """Initial learning rate."""
    lr_max: float = 1e-3
    """Maximum learning rate."""
    lr_final: float = 1e-6
    """Final learning rate."""
    n_lr_warmup: int = 143_435_655
    """Number of learning rate warmup steps."""
    wd_init = 0.04
    """Initial weight decay."""
    wd_final = 0.4
    """Final weight decay."""
    momentum_init = 0.996
    """Initial momentum."""
    momentum_final = 1.0
    """Final momentum."""

    # Evaluation
    # The typing.Annotated and tyro code makes the launch.py CLI args clean.
    newt_args: typing.Annotated[newt.Args, tyro.conf.arg(name="newt")] = (
        dataclasses.field(
            default_factory=lambda: newt.Args(
                data="/fs/scratch/PAS2136/samuelstevens/datasets/newt"
            )
        )
    )

    # Hardware
    device: str | torch.device = "cuda"
    """Hardware accelerator (if any); either 'cpu' or 'cuda'. Do not specify specific GPUs; use CUDA_VISIBLE_DEVICES for that."""
    gpus_per_node: int = 4
    """Number of GPUs per node."""
    cpus_per_task: int = 12
    """Number of CPUs per task."""
    slurm: bool = True
    """Whether to run on a slurm cluster."""
    slurm_acct: str = "PAS2136"
    """Slurm account."""

    # Misc
    wandb_entity: str = "samuelstevens"
    """WandB entity."""
    wandb_project: str = "ijepa"
    """WandB project."""
    track: bool = True
    """Whether to track runs with WandB."""
    log_every: int = 10
    """How often to log to WandB."""
    log_to: str = "logs"
    """Where to write logs."""

    # Distributed (set at runtime)
    is_ddp: bool = False
    """(set at runtime) Whether we are in a distributed setting."""
    local_rank: int = 0
    """(set at runtime) Local (current node) rank."""
    global_rank: int = 0
    """(set at runtime) Global (world) rank."""
    is_master: bool = True
    """(set at runtime) Whether this process is global master."""

    @property
    def size_p(self) -> tuple[int, int]:
        """
        Size of inputs in patches.
        """
        assert self.crop_size % self.patch_size == 0
        w_p = h_p = self.crop_size // self.patch_size
        return (w_p, h_p)


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
        list[Int[Tensor, " n_tgt_patch"]],
        Int[Tensor, " n_ctx_patch"],
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
        ctx_mask = ctx_mask.nonzero().squeeze()

        tgt_masks = torch.stack(tgt_masks)
        tgt_masks = einops.rearrange(tgt_masks, "n wp hp -> n (wp hp)")
        tgt_masks = [mask.nonzero().squeeze() for mask in tgt_masks]

        return batch, tgt_masks, ctx_mask


@beartype.beartype
def to_json_value(value: object):
    """
    Recursively converts objects into JSON-compatible values.

    As a fallback, tries to call `to_json_value()` on an object.
    """
    if value is None:
        return value

    if isinstance(value, (str, int, float)):
        return value

    if isinstance(value, (list, tuple)):
        return [to_json_value(elem) for elem in value]

    if isinstance(value, dict):
        return {to_json_value(k): to_json_value(v) for k, v in value.items()}

    if dataclasses.is_dataclass(value):
        return to_json_value(dataclasses.asdict(value))

    if isinstance(value, torch.device):
        return value.type

    raise ValueError(f"Could not convert value '{value}' to JSON-compatible value.")


def filter_no_caption_or_no_image(sample):
    has_caption = any("txt" in key for key in sample)
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


@beartype.beartype
def make_img_transform(resize_size: int, crop_size: int):
    """
    Make the image transform.

    Args:
        resize_size: How big to resize images to.
        crop_size: After resizing, how big should the crop be resized to.
    """
    img_transform = transforms.Compose([
        transforms.Resize(resize_size, antialias=True),
        transforms.RandomResizedCrop(
            crop_size,
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
    return img_transform


@beartype.beartype
def make_dataloader(args: Args, img_transform) -> torch.utils.data.DataLoader:
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


@beartype.beartype
def train(args: Args):
    ##########################
    # Distributed & Hardware #
    ##########################
    dist_env = submitit.helpers.TorchDistributedEnvironment().export()

    if args.device == "cuda":
        torch.distributed.init_process_group(backend="nccl")
        assert dist_env.rank == torch.distributed.get_rank()
        assert dist_env.world_size == torch.distributed.get_world_size()
        args = dataclasses.replace(
            args,
            device=torch.device(f"cuda:{dist_env.local_rank}"),
            is_ddp=True,
            global_rank=dist_env.rank,
            local_rank=dist_env.local_rank,
            is_master=dist_env.rank == 0,
            track=dist_env.rank == 0,
        )
    # Update eval args' device.
    args = dataclasses.replace(
        args, newt_args=dataclasses.replace(args.newt_args, device=args.device)
    )
    logger = logging.getLogger(f"Rank {args.global_rank}")
    # Debugging
    print(args)
    logger.info("%s", repr(args))

    ##########
    # Models #
    ##########
    ctx_model = modeling.VisionTransformer(
        args.d_vit,
        args.n_vit_heads,
        args.n_vit_layers,
        args.p_vit_dropout,
        args.patch_size,
        args.size_p,
    )
    # tgt_model is an exponential moving average (EMA) of ctx_model.
    tgt_model = copy.deepcopy(ctx_model)

    pred_model = modeling.PredictorTransformer(
        args.d_pred,
        args.n_pred_heads,
        args.n_pred_layers,
        args.p_pred_dropout,
        args.d_vit,
        args.size_p,
    )

    # Move models to accelerator.
    ctx_model = ctx_model.to(args.device)
    tgt_model = tgt_model.to(args.device)
    pred_model = pred_model.to(args.device)

    if args.is_ddp:
        ctx_model = torch.nn.parallel.DistributedDataParallel(
            ctx_model,
            device_ids=[args.local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        pred_model = torch.nn.parallel.DistributedDataParallel(
            pred_model,
            device_ids=[args.local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    ########
    # Data #
    ########
    img_transform = make_img_transform(args.resize_size, args.crop_size)
    dataloader = make_dataloader(args, img_transform)

    #############
    # Optimizer #
    #############
    params = [{"params": ctx_model.parameters()}, {"params": pred_model.parameters()}]
    optimizer = torch.optim.AdamW(params, fused=True)

    n_steps = args.n_imgs * args.n_epochs // args.batch_size
    lr_scheduler = schedulers.CosineWarmup(
        args.lr_init, args.lr_max, args.lr_final, args.n_lr_warmup, n_steps
    )
    wd_scheduler = schedulers.Linear(args.wd_init, args.wd_final, n_steps)
    momentum_scheduler = schedulers.Linear(
        args.momentum_init, args.momentum_final, n_steps
    )

    ###########
    # Logging #
    ###########
    config = {k: to_json_value(v) for k, v in vars(args).items()}
    mode = "online" if args.track else "disabled"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=config,
        mode=mode,
    )

    global_step = 0
    start_time = time.time()

    for epoch in range(args.n_epochs):
        logger.info("Epoch %d.", epoch + 1)

        ##############
        # Evaluation #
        ##############
        mean_acc = newt.evaluate(args.newt_args, tgt_model, img_transform)
        metrics = {"epoch": epoch, "eval/newt": mean_acc}
        run.log(metrics, step=global_step)
        logger.info("epoch: %d, step: %d, newt acc: %.5f", epoch, global_step, mean_acc)

        for b, (imgs, tgt_masks, ctx_mask) in enumerate(dataloader):
            imgs = imgs.to(args.device, non_blocking=True)
            tgt_masks = [
                tgt_mask.to(args.device, non_blocking=True) for tgt_mask in tgt_masks
            ]
            ctx_mask = ctx_mask.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                h = tgt_model(imgs)
                h = [h[:, tgt_mask, :] for tgt_mask in tgt_masks]

            z = ctx_model(imgs, ctx_mask)
            z = [pred_model(z, tgt_mask) for tgt_mask in tgt_masks]

            total_loss = torch.stack([
                F.smooth_l1_loss(z_, h_, reduction="sum") for z_, h_ in zip(z, h)
            ]).sum()
            n_patches = sum(mask.numel() for mask in tgt_masks)
            loss = total_loss / n_patches

            loss.backward()
            optimizer.step()

            # Step learning rate, momentum and weight decay.
            lr = lr_scheduler.step()
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            wd = wd_scheduler.step()
            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = wd

            with torch.no_grad():
                m = momentum_scheduler.step()
                for p_tgt, p_ctx in zip(tgt_model.parameters(), ctx_model.parameters()):
                    p_tgt.data.mul_(m).add_((1.0 - m) * p_ctx.detach().data)

            if global_step % args.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                metrics = {
                    "loss": loss.item(),
                    "learning_rate": lr,
                    "momentum": m,
                    "weight_decay": wd,
                    "epoch": epoch,
                }
                run.log(metrics, step=global_step)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
                    global_step,
                    loss.item(),
                    step_per_sec,
                )

            global_step += 1
