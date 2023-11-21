import torch
import torch.nn as nn
import torchvision.models as models


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class RPNHead(nn.Module):
    def __init__(self, in_channels):
        super(RPNHead, self).__init__()
        self.confidence = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.lengths = nn.Conv2d(in_channels, 5, kernel_size=3, padding=1)
        self.trig = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        confidence = self.sigmoid(self.confidence(x))
        lengths = self.sigmoid(self.lengths(x))
        trig = self.tanh(self.trig(x))

        # conf, x, y, l1, sin1, cos1, l2, sin2, cos2, height
        return torch.cat((confidence, lengths[:, [0, 1, 2], :, :], trig[:, [0, 1], :, :], lengths[:, [3], :, :], trig[:, [2, 3], :, :], lengths[:, [4], :, :]), dim=1)


class LiteBoxNet(nn.Module):
    def __init__(self, am, backbone_pretrained=True):
        super(LiteBoxNet, self).__init__()

        if backbone_pretrained:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            backbone = models.mobilenet_v3_small()

        attention_modules = [
            am[0](16),
            am[1](40),
            am[2](96)
        ]

        downsample_blocks = [
            nn.Sequential(*backbone.features[0:2]),
            nn.Sequential(*backbone.features[2:4]),
            nn.Sequential(*backbone.features[4:7]),
            nn.Sequential(*backbone.features[7:9]),
            nn.Sequential(*backbone.features[9:12])
        ]

        # channels_list = [576, 96, 48, 40, 24, 16, 3]
        channels_list = [576, 96, 40, 16]
        upsample_blocks = []
        for idx in range(1, len(channels_list) - 1):
            upsample_blocks.append(UpsampleBlock(channels_list[idx] * 2, channels_list[idx + 1]))

        self.attention_modules = nn.ModuleList(attention_modules)
        self.downsample_blocks = nn.ModuleList(downsample_blocks)
        self.conv_down = backbone.features[12]
        self.conv_up = nn.Conv2d(channels_list[0], channels_list[1], kernel_size=3, padding=1)
        self.upsample_blocks = nn.ModuleList(upsample_blocks)
        self.rpn_head = RPNHead(channels_list[-1])

    def freeze_backbone(self):
        for param in self.downsample_blocks.parameters():
            param.requires_grad = False
        for param in self.conv_down.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.downsample_blocks.parameters():
            param.requires_grad = True
        for param in self.conv_down.parameters():
            param.requires_grad = True

    def forward(self, x):  # 1 x 3 x 352 x 1216
        am_outputs = []
        x = self.downsample_blocks[0](x)  # 1 x 16 x 88 x 304
        am_outputs.append(self.attention_modules[0](x))  # 1 x 16 x 88 x 304
        for idx in range(1, len(self.attention_modules)):
            x = self.downsample_blocks[idx * 2 - 1](x)
            x = self.downsample_blocks[idx * 2](x)
            am_outputs.append(self.attention_modules[idx](x))

        x = self.conv_down(x)  # 1 x 576 x 11 x 38
        x = self.conv_up(x)  # 1 x 96 x 11 x 38

        am_outputs.reverse()
        for idx in range(len(self.upsample_blocks)):
            x = self.upsample_blocks[idx](torch.cat((x, am_outputs[idx]), dim=1))

        output = self.rpn_head(x)  # 1 x 10 x 44 x 152
        return output  # conf, x, y, l1, sin1, cos1, l2, sin2, cos2, height

    def get_regularization(self, weight_decay=1e-4):
        regularization = 0.0
        for param in self.parameters():
            if param.requires_grad:
                regularization += 0.5 * weight_decay * torch.sum(param**2)
        return regularization
