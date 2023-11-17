import os
import cv2
import numpy as np
from torchvision import transforms
from typing import Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset

from liteboxnet.utils.liteboxnet_utils import dict_list_to_label, read_label_file


GAUSSIAN_VALUE_7x7 = np.array([[0.0111, 0.0388, 0.0821, 0.1054, 0.0821, 0.0388, 0.0111],
                               [0.0388, 0.1353, 0.2865, 0.3679, 0.2865, 0.1353, 0.0388],
                               [0.0821, 0.2865, 0.6065, 0.7788, 0.6065, 0.2865, 0.0821],
                               [0.1054, 0.3679, 0.7788, 1.0000, 0.7788, 0.3679, 0.1054],
                               [0.0821, 0.2865, 0.6065, 0.7788, 0.6065, 0.2865, 0.0821],
                               [0.0388, 0.1353, 0.2865, 0.3679, 0.2865, 0.1353, 0.0388],
                               [0.0111, 0.0388, 0.0821, 0.1054, 0.0821, 0.0388, 0.0111]])

GAUSSIAN_VALUE_3x3 = np.array([[0.2500, 0.5000, 0.2500],
                               [0.5000, 1.0000, 0.5000],
                               [0.2500, 0.5000, 0.2500]])


class LiteBoxNetDataset(Dataset):
    def __init__(self, base_root: str, split: str, label_size: Tuple[int, int], photometric_transforms: transforms.Compose = None) -> None:
        super().__init__()

        self.photometric_transforms = photometric_transforms

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

        # self.image_files = self.image_files[:64]
        # self.label_files = self.label_files[:64]
        self.image_files.sort()
        self.label_files.sort()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        image = self.load_image(idx)
        label = self.load_label(idx, image_size=image.shape[1:])
        return image, label

    def get_meta(self, idx: int) -> Dict[str, str]:
        meta = {
            'image_name': str(os.path.basename(self.image_files[idx])),
            'label_name': str(os.path.basename(self.label_files[idx]))
        }
        return meta

    def load_image(self, idx: int) -> torch.tensor:
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        if self.photometric_transforms:
            image = self.photometric_transforms(image)
        else:
            image = transforms.ToTensor()(image)

        # image = np.transpose(image, (2, 0, 1))
        return image

    def load_label(self, idx: int, image_size: Tuple[int, int]) -> torch.tensor:

        if self.split == "testing":
            return None

        det_dict_list = read_label_file(self.label_files[idx])

        # label = dict_list_to_label(det_dict_list, image_size, self.label_size, vehicle_confidence_multiplier=np.array([[1]]))
        label = dict_list_to_label(det_dict_list, image_size, self.label_size, vehicle_confidence_multiplier=GAUSSIAN_VALUE_3x3)

        return torch.from_numpy(label)
