import torch
from torch import nn
from torch import optim
from torch.utils import data
from typing import List, Dict
from tqdm import tqdm

from liteboxnet.utils import Metric


class CustomizableTrainer(object):
    def __init__(
            self,
            device: torch.device,
            network: nn.Module,
            loss_function: nn.Module,
            optimizer: optim.Optimizer,
            train_dataloader: data.DataLoader,
            metrics: List[Metric] = [],
            with_regularization: bool = False,
            frozen_backbone: bool = True,
            show_progress: bool = False
    ) -> None:
        self.device = device
        self.network = network.to(device)
        self.loss_fn = loss_function.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.metrics = metrics
        self.with_regularization = with_regularization
        self.show_progress = show_progress
        if frozen_backbone:
            self.network.freeze_backbone()

    def step(self) -> Dict[str, float]:
        self.network.train()

        logs = {"loss": 0.0}
        for metric in self.metrics:
            logs[metric.get_name()] = 0.0

        progress_bar = None
        if self.show_progress:
            progress_bar = tqdm(total=len(self.train_dataloader), desc="Training", unit="batch")
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)

            outputs = self.network(inputs)
            loss = self.loss_fn(outputs, labels)

            if self.with_regularization:
                regularization = self.network.get_regularization()
                loss += regularization

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logs["loss"] += loss.item()
            for metric in self.metrics:
                logs[metric.get_name()] += metric(outputs, labels)

            if self.show_progress:
                progress_bar.update(1)
        if self.show_progress:
            progress_bar.close()

        logs["loss"] /= len(self.train_dataloader)
        for metric in self.metrics:
            logs[metric.get_name()] /= len(self.train_dataloader)

        return logs
