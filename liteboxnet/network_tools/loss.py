import torch
import torch.nn.functional as F


class LiteBoxNetLoss(torch.nn.Module):
    def __init__(self, conf_w: float, pos_w: float, len_w: float, trig_w: float, const_w: float):
        torch.nn.Module.__init__(self)
        self._conf_w = conf_w
        self._pos_w = pos_w
        self._len_w = len_w
        self._trig_w = trig_w
        self._const_w = const_w
        self._confidence_loss = FocalLoss(threshold=0.1)

    def forward(self, re, gt):
        mask_above_0_one_channel = gt[:, [0], :, :].ge(0).bool()
        mask_vehicle_one_channel = gt[:, [0], :, :].eq(1.0).bool()
        mask_vehicle_two_channels = mask_vehicle_one_channel.expand((mask_vehicle_one_channel.shape[0], 2, mask_vehicle_one_channel.shape[2], mask_vehicle_one_channel.shape[3]))

        gt_confidence   = gt[:, [0], :, :][mask_above_0_one_channel]
        gt_pos          = gt[:, [1, 2], :, :][mask_vehicle_two_channels]
        gt_lengths_v1   = gt[:, [3, 6], :, :][mask_vehicle_two_channels]
        gt_lengths_v2   = gt[:, [6, 3], :, :][mask_vehicle_two_channels]
        gt_sin_v1       = gt[:, [4, 7], :, :][mask_vehicle_two_channels]
        gt_sin_v2       = gt[:, [7, 4], :, :][mask_vehicle_two_channels]
        gt_cos_v1       = gt[:, [5, 8], :, :][mask_vehicle_two_channels]
        gt_cos_v2       = gt[:, [8, 5], :, :][mask_vehicle_two_channels]
        gt_height       = gt[:, [9], :, :][mask_vehicle_one_channel]

        re_confidence   = re[:, [0], :, :][mask_above_0_one_channel]
        re_pos          = re[:, [1, 2], :, :][mask_vehicle_two_channels]
        re_lengths      = re[:, [3, 6], :, :][mask_vehicle_two_channels]
        re_sin_1        = re[:, [4], :, :][mask_vehicle_one_channel]
        re_sin_2        = re[:, [7], :, :][mask_vehicle_one_channel]
        re_cos_1        = re[:, [5], :, :][mask_vehicle_one_channel]
        re_cos_2        = re[:, [8], :, :][mask_vehicle_one_channel]
        re_sin          = re[:, [4, 7], :, :][mask_vehicle_two_channels]
        re_cos          = re[:, [5, 8], :, :][mask_vehicle_two_channels]
        re_height       = re[:, [9], :, :][mask_vehicle_one_channel]

        sin_loss_v1     = F.mse_loss(re_sin, gt_sin_v1, reduction='mean')
        sin_loss_v2     = F.mse_loss(re_sin, gt_sin_v2, reduction='mean')
        cos_loss_v1     = F.mse_loss(re_cos, gt_cos_v1, reduction='mean')
        cos_loss_v2     = F.mse_loss(re_cos, gt_cos_v2, reduction='mean')
        const_1_loss    = torch.pow((1 - torch.pow(re_cos_1, 2) - torch.pow(re_sin_1, 2)), 2)
        const_1_loss    = const_1_loss.float().mean()
        const_2_loss    = torch.pow((1 - torch.pow(re_cos_2, 2) - torch.pow(re_sin_2, 2)), 2)
        const_2_loss    = const_2_loss.float().mean()

        confidence_loss = self._conf_w * self._confidence_loss(re_confidence, gt_confidence)
        pos_loss        = self._pos_w * F.smooth_l1_loss(re_pos, gt_pos, reduction='mean')
        length_loss_v1  = self._len_w * F.smooth_l1_loss(re_lengths, gt_lengths_v1, reduction='mean')
        length_loss_v2  = self._len_w * F.smooth_l1_loss(re_lengths, gt_lengths_v2, reduction='mean')
        trig_loss_v1    = self._trig_w * (sin_loss_v1 + cos_loss_v1)
        trig_loss_v2    = self._trig_w * (sin_loss_v2 + cos_loss_v2)
        const_loss      = self._const_w * (const_1_loss + const_2_loss)
        height_loss     = self._len_w * F.smooth_l1_loss(re_height, gt_height, reduction='mean')
        dimensions_loss = min(length_loss_v1 + trig_loss_v1, length_loss_v2 + trig_loss_v2) + height_loss

        print(confidence_loss.item(), pos_loss.item(), dimensions_loss.item(), const_loss.item())
        loss = confidence_loss + pos_loss + dimensions_loss + const_loss
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, threshold: float = 0.5):
        torch.nn.Module.__init__(self)
        self.threshold = threshold

    def forward(self, re, gt):
        num_pos = gt.ge(self.threshold).float().sum()
        num_neg = gt.lt(self.threshold).float().sum()

        pos_gt = gt.ge(self.threshold)
        mark_pos_loss = torch.pow(gt[pos_gt] - re[pos_gt], 2) * torch.log(re[pos_gt] + 6e-8)
        mark_pos_loss_value = -1 * mark_pos_loss.float().sum()

        neg_gt = gt.lt(self.threshold)
        mark_neg_loss = torch.pow(re[neg_gt], 2) * torch.log(1 + 6e-8 - re[neg_gt]) * torch.pow(1 - gt[neg_gt], 4)
        mark_neg_loss_value = -1 * mark_neg_loss.float().sum()

        loss = mark_neg_loss_value if num_pos == 0 else (mark_pos_loss_value + mark_neg_loss_value) / num_pos

        return loss


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, threshold: float = 0.5):
        torch.nn.Module.__init__(self)
        self.threshold = threshold

    def forward(self, re, gt):
        neg = gt.le(self.threshold).float()
        pos = gt.gt(self.threshold).float()
        weight_1 = neg.sum()
        weight_0 = pos.sum()
        mask_1 = weight_1 * pos
        mask_2 = weight_0 * neg
        mask = (mask_1 + mask_2)
        loss = (F.binary_cross_entropy(re, gt, reduction='none') * mask).float().sum() / mask.sum()

        return loss
