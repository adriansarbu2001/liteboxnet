import torch
import os
import matplotlib.pyplot as plt
from time import time
from torch import nn
from torch import optim
from liteboxnet.utils import Callback
from typing import Dict


class NetworkCheckpoint(Callback):
    def __init__(self, network: nn.Module, networks_folder: str, network_name: str):
        self.network = network
        self.networks_folder = networks_folder
        self.network_name = network_name
        self.best_loss = float("inf")
    
    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        val_loss = valid_logs.get("loss")
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                os.makedirs(self.networks_folder, exist_ok=True)
                torch.save(self.network.state_dict(), os.path.join(self.networks_folder, f"{self.network_name}.pkl"))
                print("Checkpoint saved!")


class SaveNetwork(Callback):
    def __init__(self, network: nn.Module, networks_folder: str, network_name: str):
        self.network = network
        self.networks_folder = networks_folder
        self.network_name = network_name
    
    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        os.makedirs(self.networks_folder, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(self.networks_folder, f"{self.network_name}.pkl"))
        print("Checkpoint saved!")


class PrintLogs(Callback):
    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        print("\tTraining:")
        print(f"\t\tloss: {train_logs['loss']}")
        for metric_name, metric_value in train_logs.items():
            if metric_name != "loss":
                print(f"\t\t{metric_name}: {metric_value:.4f}")
        if valid_logs is not None:
            print("\tValidation:")
            print(f"\t\tloss: {valid_logs['loss']}")
            for metric_name, metric_value in valid_logs.items():
                if metric_name != "loss":
                    print(f"\t\t{metric_name}: {metric_value:.4f}")


class ReduceLearningRate(Callback):
    def __init__(self, optimizer: optim.Optimizer, patience: int, factor: float):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.counter = 0
        self.best_loss = float("inf")
    
    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        train_loss = train_logs.get("loss")
        
        if train_loss is not None:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.counter = 0
                    print("Reducing learning rate to ", end='')
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = param_group["lr"] * self.factor
                        print(f"{param_group['lr']} ", end='')
                    print()


class EarlyStopping(Callback):
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        train_loss = train_logs.get("loss")
        if train_loss is not None:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping after {self.patience} epochs of no improvement.")
                    train_logs["stop_training"] = True


class PrintLearningRate(Callback):
    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer

    def on_epoch_start(self, epoch_idx: int) -> None:
        print(f"\n[Epoch {epoch_idx}] ", end="")
        print("Learning rate: ", end='')
        for param_group in self.optimizer.param_groups:
            print(f"{param_group['lr']} ", end='')
        print()


class PrintElapsedTime(Callback):
    def __init__(self):
        self.start = None

    def on_epoch_start(self, epoch_idx: int) -> None:
        if self.start is None:
            self.start = time()

    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        if self.start is not None:
            end = time()
            elapsed_time = end - self.start
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            print(f"Total elapsed time: {hours}:{minutes:02}:{seconds:02}")


class PlotTrainAndValidLoss(Callback):
    def __init__(self, filepath: str, start_epoch: int = 1, upper_loss: float = None):
        self.start_epoch = start_epoch
        self.filepath = filepath
        self.upper_loss = upper_loss
        self.train_losses = []
        self.valid_losses = []

    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        if epoch_idx >= self.start_epoch:
            self.train_losses.append(train_logs.get("loss"))
            self.valid_losses.append(valid_logs.get("loss"))

            epochs = range(self.start_epoch, epoch_idx + 1)

            plt.figure()

            # plt.yscale('log')
            plt.plot(epochs, self.train_losses, label='Training loss')
            plt.plot(epochs, self.valid_losses, label='Validation loss')

            plt.ylim(bottom=0.0)
            if self.upper_loss is not None:
                plt.ylim(top=self.upper_loss)

            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            plt.savefig(self.filepath)
            plt.close()


class PlotTrainLoss(Callback):
    def __init__(self, filepath: str, start_epoch: int = 1, upper_loss: float = None):
        self.start_epoch = start_epoch
        self.filepath = filepath
        self.upper_loss = upper_loss
        self.train_losses = []
        self.valid_losses = []

    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        if epoch_idx >= self.start_epoch:
            self.train_losses.append(train_logs.get("loss"))

            epochs = range(self.start_epoch, epoch_idx + 1)

            plt.figure()

            plt.plot(epochs, self.train_losses, label='Training loss')

            # plt.yscale('log')
            plt.ylim(bottom=0.0)
            if self.upper_loss is not None:
                plt.ylim(top=self.upper_loss)

            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            plt.savefig(self.filepath)
            plt.close()


class PlotMetrics(Callback):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.train_metrics = {}
        self.valid_metrics = {}

    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        if epoch_idx == 1:
            for metric_name, metric_value in train_logs.items():
                if metric_name != "loss":
                    self.train_metrics["train_" + metric_name] = []
            for metric_name, metric_value in valid_logs.items():
                if metric_name != "loss":
                    self.train_metrics["valid_" + metric_name] = []
        for metric_name, metric_value in train_logs.items():
            if metric_name != "loss":
                self.train_metrics["train_" + metric_name].append(metric_value)
        for metric_name, metric_value in valid_logs.items():
            if metric_name != "loss":
                self.train_metrics["valid_" + metric_name].append(metric_value)

        epochs = range(1, epoch_idx + 1)

        plt.figure()

        for metric_name, metric_values in self.train_metrics.items():
            plt.plot(epochs, metric_values, label=metric_name)

        plt.title('Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        plt.savefig(self.filepath)
