#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
shufflenet_v3_ext.py:
"""

import torch
import torch.nn as nn
from utils import init_params
import torch.nn.functional as F
# Variable is no longer needed in PyTorch >= 1.0; tensors track gradients natively

__author__ = "Yanjun Li"
__license__ = "MIT"


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        "Channel shuffle: [N,C,H,W] -> [N,g,C//g,H,W] -> [N,C//g,g,H,w] -> [N,C,H,W]"
        N, C, H, W, D = x.size()
        g = self.groups
        assert C % g == 0, "Number of channels ({0}) must be divisible by groups ({1})".format(C, g)
        return x.view(N, g, C // g, H, W, D).permute(0, 2, 1, 3, 4, 5).contiguous().view(N, C, H, W, D)


class BasicBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, split_channels, stride=1, split_ratio=0.5):
        """
        :param in_channels:
        :param channels_split:
        :param split_ratio: equals c'/(c+c'), where c' directly go through the block and join the next block
        """
        super(BasicBlockRes, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_channels = split_channels
        self.stride = stride
        self.split_ratio = split_ratio

        if self.split_channels:
            # Perform channel splitting
            shortcut_channels = int(self.split_ratio * self.in_channels)
            residue_channels = self.in_channels - shortcut_channels
            self.residual_residual = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(residue_channels, residue_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(residue_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=1
                nn.Conv3d(residue_channels, residue_channels, 3, 1, padding=1, groups=residue_channels, bias=False),
                nn.BatchNorm3d(residue_channels),

                # 1x1 Conv
                nn.Conv3d(residue_channels, residue_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(residue_channels)
            )

        else:
            # down-sampling, no channel-splitting
            in_channels = self.in_channels
            sub_out_channels = out_channels // 2
            self.residual_residual = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, 1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=2
                nn.Conv3d(sub_out_channels, sub_out_channels, 3, stride, padding=1, groups=sub_out_channels,
                          bias=False),
                nn.BatchNorm3d(sub_out_channels),

                # 1x1 Conv
                nn.Conv3d(sub_out_channels, sub_out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

            self.primary_brach = nn.Sequential(
                # 3x3 DWConv, stride=2
                nn.Conv3d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),

                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

        self.shuffle_block = ShuffleBlock(groups=4)  # TODO: group value

    def forward(self, x):
        if self.split_channels:
            x_primary = x[:, 0:int(self.split_ratio * self.in_channels), :, :, :]

            x_residual = x[:, int(self.split_ratio * self.in_channels):self.in_channels, :, :, :]
            x_residual_residual = self.residual_residual(x_residual)
            x_residual = F.leaky_relu(x_residual + x_residual_residual)

            x = torch.cat((x_primary, x_residual), dim=1)
            x = self.shuffle_block(x)

        else:
            x_primary = self.primary_brach(x)
            x_residual = self.residual_residual(x)
            x = torch.cat((x_primary, x_residual), dim=1)
            x = self.shuffle_block(x)
        return x


class BasicBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, split_channels, stride=1, split_ratio=0.5):
        """
        :param in_channels:
        :param channels_split:
        :param split_ratio: equals c'/(c+c'), where c' directly go through the block and join the next block
        """
        super(BasicBlockSE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_channels = split_channels
        self.stride = stride
        self.split_ratio = split_ratio

        if self.split_channels:
            # Perform channel splitting
            shortcut_channels = int(self.split_ratio * self.in_channels)
            residue_channels = self.in_channels - shortcut_channels
            self.residual_branch = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(residue_channels, residue_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(residue_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=1
                nn.Conv3d(residue_channels, residue_channels, 3, 1, padding=1, groups=residue_channels, bias=False),
                nn.BatchNorm3d(residue_channels),

                # 1x1 Conv
                nn.Conv3d(residue_channels, residue_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(residue_channels),
                nn.LeakyReLU(inplace=True),

                # SE Model
                SELayer(residue_channels)
            )

        else:
            # Don't split channel, usually down-sampling, but for 50/164, first layer of stage2 does not down-sampling
            in_channels = self.in_channels
            sub_out_channels = out_channels // 2
            self.residual_branch = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, 1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=2
                nn.Conv3d(sub_out_channels, sub_out_channels, 3, self.stride, padding=1, groups=sub_out_channels,
                          bias=False),
                nn.BatchNorm3d(sub_out_channels),

                # 1x1 Conv
                nn.Conv3d(sub_out_channels, sub_out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

            self.primary_brach = nn.Sequential(
                # 3x3 DWConv, stride=2
                nn.Conv3d(in_channels, in_channels, 3, stride=self.stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),

                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

        self.shuffle_block = ShuffleBlock(groups=4)  # TODO: group value

    def forward(self, x):
        if self.split_channels:
            x_primary = x[:, 0:int(self.split_ratio * self.in_channels), :, :, :]
            x_residual = x[:, int(self.split_ratio * self.in_channels):self.in_channels, :, :, :]
            x_residual = self.residual_branch(x_residual)
            x = torch.cat((x_primary, x_residual), dim=1)
            x = self.shuffle_block(x)

        else:
            x_primary = self.primary_brach(x)
            x_residual = self.residual_branch(x)
            x = torch.cat((x_primary, x_residual), dim=1)
            x = self.shuffle_block(x)
        return x


class BasicBlockSERes(nn.Module):
    def __init__(self, in_channels, out_channels, split_channels, stride=1, split_ratio=0.5):
        """
        :param in_channels:
        :param channels_split:
        :param split_ratio: equals c'/(c+c'), where c' directly go through the block and join the next block
        """
        super(BasicBlockSERes, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_channels = split_channels
        self.stride = stride
        self.split_ratio = split_ratio

        if self.split_channels:
            # Perform channel splitting
            shortcut_channels = int(self.split_ratio * self.in_channels)
            residue_channels = self.in_channels - shortcut_channels
            self.residual_residual = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(residue_channels, residue_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(residue_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=1
                nn.Conv3d(residue_channels, residue_channels, 3, 1, padding=1, groups=residue_channels, bias=False),
                nn.BatchNorm3d(residue_channels),

                # 1x1 Conv
                nn.Conv3d(residue_channels, residue_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(residue_channels),

                # SE Module
                SELayer(residue_channels)
            )

        else:
            # Don't split channel, usually down-sampling, but for 50/164, first layer of stage2 does not down-sampling
            in_channels = self.in_channels
            sub_out_channels = out_channels // 2
            self.residual_residual = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, 1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=2
                nn.Conv3d(sub_out_channels, sub_out_channels, 3, stride, padding=1, groups=sub_out_channels,
                          bias=False),
                nn.BatchNorm3d(sub_out_channels),

                # 1x1 Conv
                nn.Conv3d(sub_out_channels, sub_out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

            self.primary_brach = nn.Sequential(
                # 3x3 DWConv, stride=2
                nn.Conv3d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),

                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

        self.shuffle_block = ShuffleBlock(groups=4)  # TODO: group value

    def forward(self, x):
        if self.split_channels:
            x_primary = x[:, 0:int(self.split_ratio * self.in_channels), :, :, :]
            x_residual = x[:, int(self.split_ratio * self.in_channels):self.in_channels, :, :, :]
            x_residual_residual = self.residual_residual(x_residual)
            x_residual = F.leaky_relu(x_residual + x_residual_residual)

            x = torch.cat((x_primary, x_residual), dim=1)
            x = self.shuffle_block(x)

        else:
            x_primary = self.primary_brach(x)
            x_residual = self.residual_residual(x)
            x = torch.cat((x_primary, x_residual), dim=1)
            x = self.shuffle_block(x)
        return x


class InputBlock1(nn.Module):
    def __init__(self, input_channel, out_channel_config):
        super(InputBlock1, self).__init__()
        conv1_out_channels = out_channel_config[0]
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channels)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        return out


class InputBlock2(nn.Module):
    def __init__(self, input_channel, out_channel_config):
        super(InputBlock2, self).__init__()
        conv1_out_channels = out_channel_config[0]
        conv2_out_channels = out_channel_config[1]
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channels)
        self.conv2 = nn.Conv3d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(conv2_out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        return out


class OutBlock1(nn.Module):
    def __init__(self, input_channel, out_channel_config, dropout_prob):
        super(OutBlock1, self).__init__()
        self.dp1 = nn.Dropout(dropout_prob)
        conv1_out_channel = out_channel_config[0]
        self.fc1_in_channel = conv1_out_channel
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channel)
        self.dp2 = nn.Dropout(dropout_prob)
        fc1_out_channel = out_channel_config[1]
        self.fc1 = nn.Linear(conv1_out_channel, fc1_out_channel)

    def forward(self, x):
        out = self.dp1(x)
        out = F.leaky_relu(self.bn1(self.conv1(out)))
        out = self.dp2(out)
        out = out.view(out.size(0), -1, self.fc1_in_channel)
        out = self.fc1(out)
        out = out.view(out.size(0), -1)
        if self.training:
            return out
        else:
            return torch.mean(out, dim=-1, keepdim=True)


class OutBlock2(nn.Module):
    def __init__(self, input_channel, out_channel_config, dropout_prob):
        super(OutBlock2, self).__init__()
        self.dp1 = nn.Dropout(dropout_prob)
        conv1_out_channel = out_channel_config[0]
        self.fc1_in_channel = conv1_out_channel
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channel)
        self.dp2 = nn.Dropout(dropout_prob)
        fc1_out_channel = out_channel_config[1]
        self.fc1 = nn.Linear(conv1_out_channel, fc1_out_channel)
        self.ems = nn.Linear(8, 1)

    def forward(self, x):
        out = self.dp1(x)
        out = F.leaky_relu(self.bn1(self.conv1(out)))
        out = self.dp2(out)
        out = out.view(out.size(0), -1, self.fc1_in_channel)
        out = self.fc1(out)
        out1 = out.view(out.size(0), -1)
        out2 = self.ems(out1)
        if self.training:
            return torch.cat((out1, out2), -1)
        else:
            return out2


class OutBlock3(nn.Module):
    def __init__(self, input_channel, out_channel_config, dropout_prob):
        super(OutBlock3, self).__init__()
        self.dp1 = nn.Dropout(dropout_prob)
        conv1_out_channel = out_channel_config[0]
        self.fc1_in_channel = conv1_out_channel
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channel)
        self.dp2 = nn.Dropout(dropout_prob)
        fc1_out_channel = out_channel_config[1]
        self.fc1 = nn.Linear(conv1_out_channel, fc1_out_channel)

    def forward(self, x):
        out = self.dp1(x)
        out = F.leaky_relu(self.bn1(self.conv1(out)))
        out = self.dp2(out)
        out = out.view(out.size(0), -1, self.fc1_in_channel)
        if self.training:
            out = self.fc1(out)
            out = out.view(out.size(0), -1)
            return out
        else:
            out = torch.mean(out, dim=1, keepdim=False)
            return self.fc1(out)


class ShuffleNetV3Ext(nn.Module):
    def __init__(self, input_channel, dropout_prob, width_multiplier, connection):
        super(ShuffleNetV3Ext, self).__init__()
        input_block_config = [32]
        out_block_config = [2048, 1]
        width_config = {
            1.0: (116, 232, 464),
            1.5: (176, 352, 704),
            2.0: (244, 488, 976),
            50: (244, 488, 976, 1952),
            164: (340, 680, 1360, 2720)
        }
        self.width_multiplier = width_multiplier
        if self.width_multiplier in [1.0, 1.5, 2.0]:
            repeat_block = [3, 4, 4]
        elif self.width_multiplier == 50:
            repeat_block = [3, 4, 6, 3]
        elif self.width_multiplier == 164:
            repeat_block = [10, 10, 23, 10]
        channel_config = width_config[width_multiplier]
        print(channel_config, repeat_block)

        self.input_block = InputBlock1(input_channel, input_block_config)
        self.in_channels = input_block_config[-1]

        if self.width_multiplier in [1.0, 1.5, 2.0, 2.1]:
            self.stage2 = self._make_stage(channel_config[0], repeat_block[0], connection)
            self.stage3 = self._make_stage(channel_config[1], repeat_block[1], connection)
            self.stage4 = self._make_stage(channel_config[2], repeat_block[2], connection)

        elif self.width_multiplier in [50, 164]:
            self.stage2 = self._make_stage(channel_config[0], repeat_block[0], connection, first_stride=1)
            self.stage3 = self._make_stage(channel_config[1], repeat_block[1], connection)
            self.stage4 = self._make_stage(channel_config[2], repeat_block[2], connection)
            self.stage5 = self._make_stage(channel_config[3], repeat_block[3], connection)

        self.out_block = OutBlock3(channel_config[-1], out_block_config, dropout_prob)
        init_params(self)

    def _make_stage(self, out_channels, num_blocks, connection, first_stride=2):
        layers = []
        for i in range(num_blocks):
            split_channels = False if i == 0 else True
            stride = 2 if (i == 0 and first_stride == 2) else 1
            if connection == 'Res':
                layers.append(
                    BasicBlockRes(self.in_channels, out_channels, split_channels, stride=stride, split_ratio=0.5))
            elif connection == 'SE':
                layers.append(
                    BasicBlockSE(self.in_channels, out_channels, split_channels, stride=stride, split_ratio=0.5))
            elif connection == 'SERes':
                layers.append(
                    BasicBlockSERes(self.in_channels, out_channels, split_channels, stride=stride, split_ratio=0.5))
            self.in_channels = out_channels

        # dp = nn.Dropout3d(0.3)
        # layers.append(dp)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_block(x)
        if self.width_multiplier in [0.5, 1.0, 2.0, 2.1]:
            # TODO: whether keep maxpooling and stride=1 for conv1, or remove maxpooling and change stride=2 for conv1
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)
        elif self.width_multiplier in [50, 164]:
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)
            out = self.stage5(out)

        out = self.out_block(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test():
    net = ShuffleNetV3Ext(3, dropout_prob=0.5, width_multiplier=2, connection='Res')
    # net.eval()
    x = torch.randn(2, 3, 32, 32, 32)
    # x = Variable(x)  # not needed in PyTorch >= 1.0
    print(net)
    y = net(x)
    print(y)
    print(count_parameters(net))


if __name__ == '__main__':
    test()