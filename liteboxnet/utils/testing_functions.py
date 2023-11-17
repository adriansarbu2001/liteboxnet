import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader

from liteboxnet.dataset.kitti_dataset import KittiDataset
from liteboxnet.dataset.liteboxnet_dataset import LiteBoxNetDataset
from liteboxnet.network.liteboxnet import LiteBoxNet
from liteboxnet.utils.kitti_utils import plot_base_3d, plot_bboxes_3d
from liteboxnet.utils.liteboxnet_utils import plot_liteboxnet_label


def test_liteboxnet_dataset():
    photometric_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = LiteBoxNetDataset(
        base_root="D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet",
        split="training",
        label_size=(48, 156),
        photometric_transforms=photometric_transforms,
    )

    # image, label = dataset[47]
    # image, label = dataset[261]
    for idx, (image, label) in enumerate(dataset):
        # image, label = dataset[56]

        image = plot_liteboxnet_label(image=image, label=label, threshold=0.5)
        confidence = label[0, :, :].cpu().detach().numpy()

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].imshow(confidence)
        axes[1].axis('off')
        plt.tight_layout()
        # plt.savefig(os.path.join("plot_test", dataset.get_meta(idx)["image_name"]))
        plt.show()

    # plt.imshow(image)
    # plt.show()
    # plt.imshow(label[0, 0, :, :].cpu().detach().numpy())
    # plt.show()


def test_kitti_dataset():
    dataset = KittiDataset(
        base_root="D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/kitti",
        split="training"
    )

    calib, image, label, meta = dataset[8]

    image = plot_base_3d(image=image, label=label, P=calib, style="ground_truth")

    plt.imshow(image)
    plt.show()


def test_liteboxnet_dataloader():
    dataset = LiteBoxNetDataset(
        base_root="D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet",
        split="training",
        label_size=(48, 156)
    )

    loader = DataLoader(
        dataset,
        batch_size=1000,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    for batch in loader:
        images, labels = batch
        print(images.shape, labels.shape)
        pass


def test_network_output_shape(device_id=-1):
    device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)
    sample_input = torch.randn(1, 3, 370, 1224).float().to(device)
    model = LiteBoxNet(backbone_pretrained=True).to(device)
    output = model(sample_input)
    print("Output shape:", output.shape)


def test_inference(network, device_id=-1):
    device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)
    dataset = LiteBoxNetDataset(
        base_root="D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet",
        split="testing",
        label_size=(48, 156)
    )
    model = LiteBoxNet(backbone_pretrained=True).to(device)
    model.load_state_dict(torch.load(f"networks/{network}", map_location=device))
    model.eval()
    image, _ = dataset[1]
    output = model(image.unsqueeze(0).float().to(device))
    image = plot_liteboxnet_label(image, output[0], threshold=0.05)
    confidence = output[0, 0, :, :].cpu().detach().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[1].imshow(confidence)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
