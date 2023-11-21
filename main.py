import torch

from liteboxnet.network_tools import customizable_training
from liteboxnet.utils.testing_functions import test_liteboxnet_dataset, test_kitti_dataset, \
    test_network_output_shape, test_liteboxnet_dataloader, test_inference

if __name__ == "__main__":
    print("Cuda available:", torch.cuda.is_available())

    # test_liteboxnet_dataset()
    # test_liteboxnet_dataloader()
    # test_kitti_dataset()
    # test_network_output_shape(device_id=0)
    # test_inference(network="network.pkl", threshold=0.25, device_id=0)

    # label_size = (88, 304)
    # label_size = (48, 156)
    label_size = (44, 152)
    customizable_training(
        batch_size=32,
        valid_batch_size=32,
        learning_rate=1e-4,
        max_epochs=10000,
        dataset_base_root="D:/UBB/Inteligenta Computationala Aplicata EN/SEM I/datasets/liteboxnet",
        label_size=label_size,
        with_validation=True,
        device_id=0
    )
