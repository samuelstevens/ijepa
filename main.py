import logging

import dataclasses

import beartype
import tyro
import torch


log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("main")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    batch_size: int = 2048
    """Training batch size."""
    n_epochs: int = 40
    """Number of training epochs."""


@beartype.beartype
class VisionTransformer(torch.nn.Module):
    pass


@beartype.beartype
def make_dataloader(args: Args) -> torch.utils.data.DataLoader:
    dataset = 
    breakpoint()


@beartype.beartype
def main(args: Args):
    model = VisionTransformer()

    dataloader = make_dataloader(args)

    for epoch in range(args.n_epochs):
        logger.info("Epoch %d.", epoch + 1)

        for b, (imgs, masks_enc, masks_pred) in enumerate(dataloader):
            breakpoint()
        #     imgs = imgs.to(args.device, non_blocking=True)
        #     masks_enc = masks_enc.to(args.device, non_blocking=True)
        #     masks_pred = masks_pred.to(args.device, non_blocking=True)

        #     with torch.no_grad():
        #         h = tgt_enc(imgs)
        #         h = F.layer_norm(h, (h.size(-1),))  # normalize over feature dimension
        #         B = len(h)
        #         h = apply_masks(h, masks_pred)
        #         h = repeat_interleave_batch(h, B, repeat=len(masks_enc))


if __name__ == "__main__":
    main(tyro.cli(Args))
