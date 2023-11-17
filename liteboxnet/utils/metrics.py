import torch
from liteboxnet.utils import Metric


class TP(Metric):
    def __init__(self, threshold: float = 0.5):
        super(TP, self).__init__()
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        re_confidence = outputs[:, 0, :, :]
        gt_confidence = labels[:, 0, :, :]
        mask_above_0 = gt_confidence.ge(0).bool()
        re_confidence = re_confidence[mask_above_0]
        gt_confidence = gt_confidence[mask_above_0]
        tp = torch.logical_and(re_confidence.gt(self.threshold), gt_confidence.eq(1.0)).float().sum()
        return tp.item()
    
    def get_name(self) -> str:
        return "True Positive"


class FP(Metric):
    def __init__(self, threshold: float = 0.5):
        super(FP, self).__init__()
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        re_confidence = outputs[:, 0, :, :]
        gt_confidence = labels[:, 0, :, :]
        mask_above_0 = gt_confidence.ge(0).bool()
        re_confidence = re_confidence[mask_above_0]
        gt_confidence = gt_confidence[mask_above_0]
        fp = torch.logical_and(re_confidence.gt(self.threshold), gt_confidence.eq(0.0)).float().sum()
        return fp.item()

    def get_name(self) -> str:
        return "False Positive"


class FN(Metric):
    def __init__(self, threshold: float = 0.5):
        super(FN, self).__init__()
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        re_confidence = outputs[:, 0, :, :]
        gt_confidence = labels[:, 0, :, :]
        mask_above_0 = gt_confidence.ge(0).bool()
        re_confidence = re_confidence[mask_above_0]
        gt_confidence = gt_confidence[mask_above_0]
        fn = torch.logical_and(re_confidence.le(self.threshold), gt_confidence.eq(1.0)).float().sum()
        return fn.item()

    def get_name(self) -> str:
        return "False Negative"


class Precision(Metric):
    def __init__(self, threshold: float = 0.5):
        super(Precision, self).__init__()
        self.threshold = threshold
        self.tp = TP(threshold=threshold)
        self.fp = FP(threshold=threshold)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        tp = self.tp(outputs, labels)
        fp = self.fp(outputs, labels)
        if tp + fp == 0:
            return 1.0
        else:
            return tp / (tp + fp)

    def get_name(self) -> str:
        return "Precision"


class Recall(Metric):
    def __init__(self, threshold: float = 0.5):
        super(Recall, self).__init__()
        self.threshold = threshold
        self.tp = TP(threshold=threshold)
        self.fn = FN(threshold=threshold)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        tp = self.tp(outputs, labels)
        fn = self.fn(outputs, labels)
        if tp + fn == 0:
            return 1.0
        else:
            return tp / (tp + fn)

    def get_name(self) -> str:
        return "Recall"


class F1Score(Metric):
    def __init__(self, threshold: float = 0.5):
        super(F1Score, self).__init__()
        self.threshold = threshold
        self.tp = TP(threshold=threshold)
        self.fp = FP(threshold=threshold)
        self.fn = FN(threshold=threshold)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        tp = self.tp(outputs, labels)
        fp = self.fp(outputs, labels)
        fn = self.fn(outputs, labels)
        if tp + 0.5 * (fp + fn) == 0:
            return 1.0
        else:
            return tp / (tp + 0.5 * (fp + fn))

    def get_name(self) -> str:
        return "F1 score"
