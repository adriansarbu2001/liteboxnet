from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple

from liteboxnet.dataset.liteboxnet_dataset import LiteBoxNetDataset


def get_dataloader(base_root: str, split: str, batch_size: int, label_size: Tuple[int, int], photometric_transforms: transforms.Compose = None):
    shuffle = False
    if split == "training":
        shuffle = True

    dataset = LiteBoxNetDataset(
        base_root=base_root,
        split=split,
        label_size=label_size,
        photometric_transforms=photometric_transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle,
        drop_last=True,
    )
    return dataloader
