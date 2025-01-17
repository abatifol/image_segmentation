import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.layer(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class EncoderBlockSegRes(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, stride=1, padding=1):
        super(EncoderBlockSegRes, self).__init__()
        self.residual_layers = nn.ModuleList()
        for i in range(depth):
            self.residual_layers.append(ResidualBlock(in_channels if i == 0 else out_channels, out_channels, stride, padding))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        for layer in self.residual_layers:
            x = layer(x)
        down, indices = self.pool(x)
        return down, x, indices  # Down for next layer, x for skip connection
    

class DecoderBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, kernel_size=3, padding=1, classification=False) -> None:
        super(DecoderBlockSkip, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:  # Adjust the first layer for concatenated channels
                self.layers.append(ConvBlock(in_channels * 2, in_channels, kernel_size=kernel_size, padding=padding))
            elif i == depth - 1 and classification:
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            elif i == depth - 1:
                self.layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            else:
                self.layers.append(ConvBlock(in_channels, in_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x, ind, skip):
        x = self.unpool(x, ind)
        # print(f"decoder unpool: {x.shape}, skip :{skip.shape}")
        x = torch.cat([x, skip], 1)
        for layer in self.layers:
            x = layer(x)
        return x

class ResidualUSegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualUSegNet, self).__init__()
        # Encoder
        self.enc1 = EncoderBlockSegRes(in_channels, 64, depth=2)
        self.enc2 = EncoderBlockSegRes(64, 128, depth=2)
        self.enc3 = EncoderBlockSegRes(128, 256, depth=3)


        # Decoder
        self.dec3 = DecoderBlockSkip(256, 128, depth=3)
        self.dec2 = DecoderBlockSkip(128, 64, depth=2)
        self.dec1 = DecoderBlockSkip(64, out_channels, depth=2)


    def forward(self, x):
        # Encoder
        x1, skip1, ind1 = self.enc1(x)
        x2, skip2, ind2 = self.enc2(x1)
        x3, skip3, ind3 = self.enc3(x2)

        # Decoder
        d3 = self.dec3(x3, ind3, skip3)
        d2 = self.dec2(d3, ind2, skip2)
        d1 = self.dec1(d2, ind1, skip1)
        return d1
