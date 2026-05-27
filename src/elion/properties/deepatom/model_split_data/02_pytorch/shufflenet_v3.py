#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
shufflenet_v3.py:
"""
import torch
import torch.nn as nn
from utils import init_params
import torch.nn.functional as F
# Variable is no longer needed in PyTorch >= 1.0; tensors track gradients natively

__author__ = "Yanjun Li"
__license__ = "MIT"


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        "Channel shuffle: [N,C,H,W] -> [N,g,C//g,H,W] -> [N,C//g,g,H,w] -> [N,C,H,W]"
        N, C, H, W, D = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W,D).permute(0,2,1,3,4,5).contiguous().view(N,C,H,W,D)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, split_ratio=0.5):
        """
        :param in_channels:
        :param channels_split:
        :param split_ratio: equals c'/(c+c'), where c' directly go through the block and join the next block
        """
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.split_ratio = split_ratio

        if self.stride == 1:
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
                nn.LeakyReLU(inplace=True)
            )

        else:
            # down-sampling, no channel-splitting
            in_channels = self.in_channels
            sub_out_channels = out_channels // 2
            self.residual_branch = nn.Sequential(
                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, 1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True),

                # 3x3 DWConv, stride=2
                nn.Conv3d(sub_out_channels, sub_out_channels, 3, 2, padding=1, groups=sub_out_channels, bias=False),
                nn.BatchNorm3d(sub_out_channels),

                # 1x1 Conv
                nn.Conv3d(sub_out_channels, sub_out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

            self.primary_brach = nn.Sequential(
                # 3x3 DWConv, stride=2
                nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm3d(in_channels),

                # 1x1 Conv
                nn.Conv3d(in_channels, sub_out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(sub_out_channels),
                nn.LeakyReLU(inplace=True)
            )

        self.shuffle_block = ShuffleBlock(groups=4)       # TODO: group value

    def forward(self, x):
        if self.stride == 1:
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


class InputBlock1(nn.Module):
    """1 Conv layer and 1 down-sampling max pooling layer"""
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
    """ (Not used) 2 Conv layers and the second one down-samples."""
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


class InputBlock3(nn.Module):
    def __init__(self, input_channel, out_channel_config):
        """(Not use): 2 Conv layers and 1 max-pooling layer. Down sampling: 2 times"""
        super(InputBlock3, self).__init__()
        conv1_out_channels = out_channel_config[0]
        conv2_out_channels = out_channel_config[1]
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channels)
        self.conv2 = nn.Conv3d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(conv2_out_channels)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.maxpool1(out)    
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
    """1 Conv, 1 FC, Training: 2x2x2 outputs; Testing: first averaging then output 1 after FC (Used for 32 grid size)"""
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


class OutBlock4(nn.Module):
    """1 Conv, 1 FC, adaptive_avg_pooling to make sure training: 2x2x2 outputs;
    Testing: first averaging then output 1 after FC
    (Used for all grid sizes)"""
    def __init__(self, input_channel, out_channel_config, dropout_prob):
        super(OutBlock4, self).__init__()
        self.dp1 = nn.Dropout(dropout_prob)
        conv1_out_channel = out_channel_config[0]
        self.fc1_in_channel = conv1_out_channel
        self.conv1 = nn.Conv3d(input_channel, conv1_out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(conv1_out_channel)
        self.dp2 = nn.Dropout(dropout_prob)
        self.avg_pool1 = nn.AdaptiveAvgPool3d(2)
        fc1_out_channel = out_channel_config[1]
        self.fc1 = nn.Linear(conv1_out_channel, fc1_out_channel)

    def forward(self, x):
        out = self.dp1(x)
        out = F.leaky_relu(self.bn1(self.conv1(out)))
        out = self.dp2(out)
        out = self.avg_pool1(out)
        out = out.view(out.size(0), -1, self.fc1_in_channel)
        if self.training:
            out = self.fc1(out)
            out = out.view(out.size(0), -1)
            return out
        else:
            out = torch.mean(out, dim=1, keepdim=False)
            return self.fc1(out)


class ShuffleNetV3(nn.Module):
    def __init__(self, input_channel, dropout_prob, width_multiplier):
        super(ShuffleNetV3, self).__init__()
        input_block_config = [32]         # [32]
        out_block_config = [2048, 1]          # [1]
        width_config = {
            0.25: (24, 48, 96),
            0.33: (32, 64, 128),
            0.5: (48, 96, 192),
            1.0: (116, 232, 464),
            1.5: (176, 352, 704),
            2.0: (244, 488, 976),
        }
        repeat_block = [3, 4, 4]
        channel_config = width_config[width_multiplier]
        print('channel_config: %s' % str(channel_config))
        print('repeat_block: %s' % str(repeat_block))

        self.input_block = InputBlock1(input_channel, input_block_config)
        self.in_channels = input_block_config[-1]

        self.stage2 = self._make_stage(channel_config[0], repeat_block[0])
        self.stage3 = self._make_stage(channel_config[1], repeat_block[1])
        self.stage4 = self._make_stage(channel_config[2], repeat_block[2])

        self.out_block = OutBlock4(channel_config[-1], out_block_config, dropout_prob)
        init_params(self)

    def _make_stage(self, out_channel, num_blocks):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(BasicBlock(self.in_channels, out_channel, stride=stride, split_ratio=0.5))
            self.in_channels = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_block(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.out_block(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test():
    net = ShuffleNetV3(3, dropout_prob=0.5, width_multiplier=2.0)
    x = torch.randn(1, 3, 86, 86, 86)
    # x = Variable(x)  # not needed in PyTorch >= 1.0
    y = net(x)
    print(y)
    print(count_parameters(net))


if __name__ == '__main__':
    test()