import torch
from torch import nn


class PAM(nn.Module):
    """ Position Attention Module """

    def __init__(self, channel):
        super(PAM, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        _, c, _, _ = x.size()
        y = self.act(self.conv(x))
        y = y.repeat(1, c, 1, 1)
        return x * y


class CAM(nn.Module):
    """ Channel Attention Module """

    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DAMPosition(nn.Module):
    """ Position attention submodule in Dual Attention Module"""

    # Ref from https://github.com/junfu1115/DANet
    def __init__(self, in_dim):
        super(DAMPosition, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DAMChannel(nn.Module):
    """ Channel attention submodule in Dual Attention Module """

    # Ref from https://github.com/junfu1115/DANet
    def __init__(self, in_dim):
        super(DAMChannel, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        return out


class DAM(nn.Module):
    """ Dual Attention Module """

    def __init__(self, in_dim):
        super(DAM, self).__init__()

        self.dam_p = DAMPosition(in_dim)
        self.dam_c = DAMChannel(in_dim)

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        out = self.dam_p(x) + self.dam_c(x)
        return out
