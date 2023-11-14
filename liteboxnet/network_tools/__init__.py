import torch
from torch.backends import cudnn
from typing import List

from liteboxnet.dataset import get_dataloader
from liteboxnet.network.liteboxnet import LiteBoxNet
from liteboxnet.network_tools.loss import LiteBoxNetLoss
from liteboxnet.network_tools.trainer import CustomizableTrainer
from liteboxnet.network_tools.validator import CustomizableValidator
from liteboxnet.utils import Callback, Metric
from liteboxnet.utils.callbacks import PrintLearningRate, PrintLogs, ReduceLearningRate, SaveNetwork, PlotTrainLoss, \
    PrintElapsedTime, EarlyStopping, PlotTrainAndValidLoss
from liteboxnet.utils.metrics import Precision, Recall, F1Score


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def customizable_training(batch_size, valid_batch_size, learning_rate, max_epochs, dataset_base_root, label_size, with_validation=True, device_id=-1):
    # Initial
    setup(19960229)
    device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)

    # Network
    network = LiteBoxNet(backbone_pretrained=True)
    networks_folder = "networks"
    network_name = "network"

    # Dataset
    train_dataloader = get_dataloader(
        base_root=dataset_base_root,
        split="training",
        batch_size=batch_size,
        label_size=label_size
    )
    valid_dataloader = None
    if with_validation:
        valid_dataloader = get_dataloader(
            base_root=dataset_base_root,
            split="validating",
            batch_size=valid_batch_size,
            label_size=label_size
        )

    # Loss and optimizer
    loss_function = LiteBoxNetLoss(conf_w=1, pos_w=1, len_w=1, trig_w=1, const_w=4)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # Callbacks and Metrics
    callbacks: List[Callback] = [
        PrintLearningRate(optimizer=optimizer),
        PrintLogs(),
        ReduceLearningRate(optimizer=optimizer, patience=15, factor=5e-1),
        SaveNetwork(network=network, networks_folder=networks_folder, network_name=network_name),
        PlotTrainAndValidLoss(f"plots/{network_name}_loss_evolution.png", start_epoch=1, upper_loss=1.0),
        PrintElapsedTime(),
        EarlyStopping(patience=45),
    ]
    threshold = 0.5
    metrics: List[Metric] = [
        Precision(threshold=threshold),
        Recall(threshold=threshold),
        F1Score(threshold=threshold),
    ]

    # Trainer and Validator
    trainer = CustomizableTrainer(
        device=device,
        network=network,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        metrics=metrics,
        with_regularization=False,
        frozen_backbone=True,
        show_progress=True
    )
    validator = None
    if with_validation:
        validator = CustomizableValidator(
            device=device,
            network=network,
            loss_function=loss_function,
            valid_dataloader=valid_dataloader,
            metrics=metrics,
            with_regularization=False,
            show_progress=True
        )

    # Training loop
    for epoch in range(1, max_epochs + 1):

        for callback in callbacks:
            callback.on_epoch_start(epoch_idx=epoch)

        train_logs = trainer.step()
        valid_logs = {}
        if with_validation:
            valid_logs = validator.step()

        for callback in callbacks:
            callback.on_epoch_end(epoch_idx=epoch, train_logs=train_logs, valid_logs=valid_logs)

        if "stop_training" in train_logs and train_logs["stop_training"] == True:
            break
