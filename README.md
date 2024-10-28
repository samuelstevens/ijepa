# I-JEPA

I-JEPA from scratch for fun.

Training a tiny ViT with `uv` on a Slurm cluster:

```sh
uv run launch.py --batch-size 1024 --gpus-per-node 4 --d-vit 192 --n-vit-heads 3 --n-vit-layers 12 --p-vit-dropout 0.0 --d-pred 96 --n-pred-heads 3 --n-pred-layers 6 --p-pred-dropout 0.0
```
