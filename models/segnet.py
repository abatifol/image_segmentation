
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

class EncoderBlockSegnet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, kernel_size=3, padding=1):
        super(EncoderBlockSegnet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ConvBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x, indices = self.pool(x)
        return x, indices

class DecoderBlockSegnet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, kernel_size=3, padding=1, classification=False) -> None:
        super(DecoderBlockSegnet, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == depth - 1 and classification:
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            elif i == depth - 1:
                self.layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            else:
                self.layers.append(ConvBlock(in_channels, in_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x, ind):
        x = self.unpool(x, ind)
        for layer in self.layers:
            x = layer(x)
        return x

class SegNet(nn.Module): 
    def __init__(self, in_channels=3, out_channels=1, features=64) -> None:
        super(SegNet, self).__init__()

        # Encoder
        self.enc0 = EncoderBlockSegnet(in_channels, features, depth=2)
        self.enc1 = EncoderBlockSegnet(features, features * 2, depth=2)
        self.enc2 = EncoderBlockSegnet(features * 2, features * 4, depth=3)
        self.enc3 = EncoderBlockSegnet(features * 4, features * 8, depth=3)
        self.enc4 = EncoderBlockSegnet(features * 8, features * 8, depth=3)

        # Decoder
        self.dec0 = DecoderBlockSegnet(features * 8, features * 8, depth=3) 
        self.dec1 = DecoderBlockSegnet(features * 8, features * 4, depth=3)
        self.dec2 = DecoderBlockSegnet(features * 4, features * 2, depth=3)
        self.dec3 = DecoderBlockSegnet(features * 2, features, depth=2)
        self.dec4 = DecoderBlockSegnet(features, out_channels, depth=2, classification=True) # No activation

    def forward(self, x):
        # encoder
        e0, ind0 = self.enc0(x) 
        e1, ind1 = self.enc1(e0) 
        e2, ind2 = self.enc2(e1) 
        e3, ind3 = self.enc3(e2)
        e4, ind4 = self.enc4(e3)

        # decoder
        d0 = self.dec0(e4, ind4)
        d1 = self.dec1(d0, ind3)
        d2 = self.dec2(d1, ind2)
        d3 = self.dec3(d2, ind1)

        # classification layer
        output = self.dec4(d3, ind0)  
        # return F.sigmoid(output)
        return output
