import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResidualUNet(nn.Module):
    def __init__(self, in_channels, out_channels,filters=[64, 128, 256, 512]):
        super(ResidualUNet, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)

        self.enc2 = ResidualBlock(filters[0], filters[1], 2, 1)
        self.enc3 = ResidualBlock(filters[1], filters[2], 2, 1)

        self.bridge = ResidualBlock(filters[2], filters[3], 2, 1)

        self.upsample1 = Upsample(filters[3], filters[3], 2, 2)
        self.dec1 = ResidualBlock(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample2 = Upsample(filters[2], filters[2], 2, 2)
        self.dec2 = ResidualBlock(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample3 = Upsample(filters[1], filters[1], 2, 2)
        self.dec3 = ResidualBlock(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 1, 1)
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoders
        down1 = self.input_conv(x) + self.input_skip(x)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)

        # Bridge
        b = self.bridge(down3)

        # Decoders
        up1 = self.upsample1(b)
        up1 = torch.cat([up1, down3], dim=1)
        up1 = self.dec1(up1)

        up2 = self.upsample2(up1)
        up2 = torch.cat([up2, down2], dim=1)
        up2 = self.dec2(up2)

        up3 = self.upsample3(up2)
        up3 = torch.cat([up3, down1], dim=1)
        up3 = self.dec3(up3)

        output = self.output_layer(up3)

        return output