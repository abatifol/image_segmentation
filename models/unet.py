
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DecoderBlockUnet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels,out_channels)

    def forward(self,x,skip):
        x = self.up(x)
        return self.double_conv(torch.cat([x,skip],1))

class EncoderBlockUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.double_conv(x)
        p = self.pool(down)

        return down, p

class UNet(nn.Module):
    def __init__(self,in_channels=1,out_channels=1):
        super(UNet,self).__init__()
        self.enc1 = EncoderBlockUnet(in_channels,64)
        self.enc2 = EncoderBlockUnet(64,128)
        self.enc3 = EncoderBlockUnet(128,256)
        self.enc4 = EncoderBlockUnet(256,512)
        self.bottleneck = DoubleConv(512,1024)

        self.dec1 = DecoderBlockUnet(1024,512)
        self.dec2 = DecoderBlockUnet(512,256)
        self.dec3 = DecoderBlockUnet(256,128)
        self.dec4 = DecoderBlockUnet(128,64)

        self.final_conv = nn.Conv2d(64,out_channels, kernel_size=1)

    def forward(self, x):
        down1, x1 = self.enc1(x)
        down2, x2 = self.enc2(x1)
        down3, x3 = self.enc3(x2)
        down4, x4 = self.enc4(x3)

        bottleneck = self.bottleneck(x4)
        # print(f"Bottleneck shape: {bottleneck.shape} enc4 shape: {x4.shape}")
        dec1 = self.dec1(bottleneck,down4)
        dec2 = self.dec2(dec1, down3)
        dec3 = self.dec3(dec2, down2)
        dec4 = self.dec4(dec3, down1)

        # return torch.sigmoid(self.final_conv(dec4))
        return self.final_conv(dec4)
