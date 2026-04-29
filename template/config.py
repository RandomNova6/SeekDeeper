import torch

from pathlib import Path

torch.manual_seed(3407)

# training setting
batch_size = ...
lr = ...


# path
base_dir = Path(__file__).parent.resolve()
checkpoint_dir = base_dir / "checkpoints"
dataset_dir = base_dir / "datasets"

# inference
