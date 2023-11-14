import torch
from torch import nn
from typing import List, Dict
from torch.utils import data
from tqdm import tqdm

from liteboxnet.utils import Metric


class CustomizableValidator(object):
    def __init__(
            self,
            device: torch.device,
            network: nn.Module,
            loss_function: nn.Module,
            valid_dataloader: data.DataLoader,
            metrics: List[Metric] = [],
            with_regularization: bool = False,
            show_progress: bool = False
    ) -> None:
        self.device = device
        self.network = network.to(device)
        self.loss_fn = loss_function.to(device)
        self.valid_dataloader = valid_dataloader
        self.metrics = metrics
        self.with_regularization = with_regularization
        self.show_progress = show_progress

    def step(self) -> Dict[str, float]:
        self.network.eval()

        logs = {"loss": 0.0}
        for metric in self.metrics:
            logs[metric.get_name()] = 0.0

        with torch.no_grad():
            progress_bar = None
            if self.show_progress:
                progress_bar = tqdm(total=len(self.valid_dataloader), desc="Validating", unit="batch")
            for inputs, labels in self.valid_dataloader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)

                outputs = self.network(inputs)
                loss = self.loss_fn(outputs, labels)

                if self.with_regularization:
                    regularization = self.network.get_regularization()
                    loss += regularization

                logs["loss"] += loss.item()
                for metric in self.metrics:
                    logs[metric.get_name()] += metric(outputs, labels)

                if self.show_progress:
                    progress_bar.update(1)
            if self.show_progress:
                progress_bar.close()

        logs["loss"] /= len(self.valid_dataloader)
        for metric in self.metrics:
            logs[metric.get_name()] /= len(self.valid_dataloader)

        return logs
