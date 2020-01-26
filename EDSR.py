import torch
from torch import nn
from torchsummary import summary

n_feat = 256
kernel_size = 3


class _Res_Block(nn.Module):
    def __init__(self):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        y = self.relu(self.res_conv(x))
        y = self.res_conv(y)
        y *= 0.1
        y = torch.add(y, x)
        return y


class edsr(nn.Module):
    def __init__(self):
        super(edsr, self).__init__()

        in_ch = 3
        num_blocks = 32

        self.conv1 = nn.Conv2d(in_ch, n_feat, kernel_size, padding=1)
        self.conv_up = nn.Conv2d(n_feat, n_feat * 4, kernel_size, padding=1)
        self.conv_out = nn.Conv2d(n_feat, in_ch, kernel_size, padding=1)

        self.body = self.make_layer(_Res_Block, num_blocks)

        self.upsample = nn.Sequential(self.conv_up, nn.PixelShuffle(2))

    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        return nn.Sequential(*res_block)

    def forward(self, x):

        out = self.conv1(x)
        out = self.body(out)
        out = self.upsample(out)
        out = self.conv_out(out)

        return out
