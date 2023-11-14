import os
import cv2
import numpy as np
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset

from liteboxnet.utils.liteboxnet_utils import dict_list_to_label, read_label_file


class LiteBoxNetDataset(Dataset):
    def __init__(self, base_root: str, split: str, label_size: Tuple[int, int]) -> None:
        super().__init__()

        base_root = os.path.abspath(base_root)
        assert os.path.isdir(base_root)
        self.base_root = base_root

        assert split in ['training', 'validating', 'testing']
        self.split = split

        # Image Files
        self.image_dir = os.path.join(base_root, split, 'image_2')
        self.image_files = [os.path.join(self.image_dir, image) for image in os.listdir(self.image_dir)]

        # Label Files
        self.label_size = label_size
        self.label_dir = None
        self.label_files = []
        if split != 'testing':
            self.label_dir = os.path.join(base_root, split, 'label_2')
            self.label_files = [os.path.join(self.label_dir, label) for label in os.listdir(self.label_dir)]

        self.image_files.sort()
        self.label_files.sort()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(idx)
        label = self.load_label(idx, image_size=image.shape[1:])
        return image, label

    def get_meta(self, idx: int) -> Dict[str, str]:
        meta = {
            'image_name': str(os.path.basename(self.image_files[idx])),
            'label_name': str(os.path.basename(self.label_files[idx]))
        }
        return meta

    def load_image(self, idx: int) -> np.ndarray:
        image_arr = cv2.imread(self.image_files[idx])
        image_data = cv2.cvtColor(image_arr, code=cv2.COLOR_BGR2RGB)
        image_data = np.transpose(image_data, (2, 0, 1)) / 255.0

        return torch.from_numpy(image_data.astype(float))

    def load_label(self, idx: int, image_size: Tuple[int, int]) -> np.ndarray:
        if self.split == "testing":
            return None

        det_dict_list = read_label_file(self.label_files[idx])
        label = dict_list_to_label(det_dict_list, image_size, self.label_size)

        return torch.from_numpy(label)
