import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT) if pretrained else models.mobilenet_v3_small()
        # self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.features = self.backbone.features

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        return x


class RPNHead(nn.Module):
    def __init__(self, in_channels):
        super(RPNHead, self).__init__()
        # conf, x, y, l1, sin1, cos1, l2, sin2, cos2, height
        self.confidence = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.lengths = nn.Conv2d(in_channels, 5, kernel_size=3, padding=1)
        self.trig = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        confidence = self.sigmoid(self.confidence(x))
        lengths = self.sigmoid(self.lengths(x))
        trig = self.tanh(self.trig(x))
        return torch.cat((confidence, lengths[:, [0, 1, 2], :, :], trig[:, [0, 1], :, :], lengths[:, [3], :, :], trig[:, [2, 3], :, :], lengths[:, [4], :, :]), dim=1)


class LiteBoxNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(LiteBoxNet, self).__init__()
        self.backbone = ResNetBackbone(pretrained=backbone_pretrained)
        # self.neck = Decoder(1280, 256)  # for mobilenetv2
        self.neck = Decoder(576, 256)  # for mobilenetv3_small
        self.rpn_head = RPNHead(256)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        # print(x.shape)
        resnet_features = self.backbone(x)
        # print(resnet_features.shape)
        fpn_features = self.neck(resnet_features)
        # print(fpn_features.shape)
        output = self.rpn_head(fpn_features)
        # print(regression.shape)
        return output
