import torch
from typing import Dict


class Callback:
    def on_epoch_start(self, epoch_idx: int) -> None:
        pass
    
    def on_epoch_end(self, epoch_idx: int, train_logs: Dict[str, float], valid_logs: Dict[str, float]) -> None:
        pass


class Metric:
    def __init__(self):
        pass
    
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        raise NotImplementedError("Metric function must be implemented in subclasses")
    
    def get_name(self) -> str:
        return "NO_DEFINED_NAME"
