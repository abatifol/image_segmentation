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

class EncoderBlockUsegnet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, kernel_size=3, padding=1):
        super(EncoderBlockUsegnet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ConvBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, down):
        for layer in self.layers:
            down = layer(down)
        x, indices = self.pool(down)
        return x, down, indices #down for skip connection


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
                self.layers.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x, ind):
        x = self.unpool(x, ind)
        for layer in self.layers:
            x = layer(x)
        return x

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


class SegUNet_3skip(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SegUNet_3skip, self).__init__()
        self.enc1 = EncoderBlockUsegnet(in_channels, 64, depth=2)
        self.enc2 = EncoderBlockUsegnet(64, 128, depth=2)
        self.enc3 = EncoderBlockUsegnet(128, 256, depth=3)
        # self.enc4 = EncoderBlock(256,256, depth=3)

        # self.dec2 = DecoderBlock(256,256,depth=3)
        self.dec3 = DecoderBlockSkip(256,128, depth=3)
        self.dec4 = DecoderBlockSkip(128,64,depth=2)
        self.dec5 = DecoderBlockSkip(64,out_channels, depth=2, classification=True)  # Classification=True ensures correct final output.

        # self_final = nn.Conv2d(64,out_channels, kernel_size=1)    
    def forward(self, x):
        # Encoder
        x1, down1, ind1 = self.enc1(x)
        # print(f"x1: {x1.shape}, down1: {down1.shape}, ind1: {ind1.shape}")
        x2, down2, ind2 = self.enc2(x1)
        # print(f"x2: {x2.shape}, down2: {down2.shape}, ind2: {ind2.shape}")
        x3, down3, ind3 = self.enc3(x2)
        # print(f"x3: {x3.shape}, down3: {down3.shape}, ind3: {ind3.shape}")
        # x4, down4, ind4 = self.enc4(x3)
        # print(f"x4: {x4.shape}, down4: {down4.shape}, ind4: {ind4.shape}")

        # Decoder
        # dec2 = self.dec2(x4,ind4,down4)
        dec3 = self.dec3(x3, ind3, down3)
        dec4 = self.dec4(dec3, ind2, down2)
        dec5 = self.dec5(dec4, ind1, down1)

        return dec5

class SegUNet_2skip(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SegUNet_2skip, self).__init__()
        self.enc1 = EncoderBlockUsegnet(in_channels, 64, depth=2)
        self.enc2 = EncoderBlockUsegnet(64, 128, depth=2)
        self.enc3 = EncoderBlockUsegnet(128, 256, depth=3)
        # self.enc4 = EncoderBlock(256,256, depth=3)

        # self.dec2 = DecoderBlock(256,256,depth=3)
        self.dec3 = DecoderBlockSegnet(256,128, depth=3)
        self.dec4 = DecoderBlockSkip(128,64,depth=2)
        self.dec5 = DecoderBlockSkip(64,out_channels, depth=2, classification=True)  # Classification=True ensures correct final output.
  
    def forward(self, x):
        # Encoder
        x1, down1, ind1 = self.enc1(x)
        # print(f"x1: {x1.shape}, down1: {down1.shape}, ind1: {ind1.shape}")
        x2, down2, ind2 = self.enc2(x1)
        # print(f"x2: {x2.shape}, down2: {down2.shape}, ind2: {ind2.shape}")
        x3, down3, ind3 = self.enc3(x2)
        # print(f"x3: {x3.shape}, down3: {down3.shape}, ind3: {ind3.shape}")
        # x4, down4, ind4 = self.enc4(x3)
        # print(f"x4: {x4.shape}, down4: {down4.shape}, ind4: {ind4.shape}")

        # Decoder
        # dec2 = self.dec2(x4,ind4,down4)
        dec3 = self.dec3(x3, ind3) #, down3)
        dec4 = self.dec4(dec3, ind2, down2)
        dec5 = self.dec5(dec4, ind1, down1)

        return dec5


class SegUNet_1skip(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SegUNet_1skip, self).__init__()
        self.enc1 = EncoderBlockUsegnet(in_channels, 64, depth=2)
        self.enc2 = EncoderBlockUsegnet(64, 128, depth=2)
        self.enc3 = EncoderBlockUsegnet(128, 256, depth=3)
        # self.enc4 = EncoderBlock(256,256, depth=3)

        # self.dec2 = DecoderBlock(256,256,depth=3)
        self.dec3 = DecoderBlockSegnet(256,128, depth=3)
        self.dec4 = DecoderBlockSegnet(128,64,depth=2)
        self.dec5 = DecoderBlockSkip(64,out_channels, depth=2, classification=True)  # Classification=True ensures correct final output.
   
    def forward(self, x):
        # Encoder
        x1, down1, ind1 = self.enc1(x)
        # print(f"x1: {x1.shape}, down1: {down1.shape}, ind1: {ind1.shape}")
        x2, down2, ind2 = self.enc2(x1)
        # print(f"x2: {x2.shape}, down2: {down2.shape}, ind2: {ind2.shape}")
        x3, down3, ind3 = self.enc3(x2)
        # print(f"x3: {x3.shape}, down3: {down3.shape}, ind3: {ind3.shape}")
        # x4, down4, ind4 = self.enc4(x3)
        # print(f"x4: {x4.shape}, down4: {down4.shape}, ind4: {ind4.shape}")

        # Decoder
        # dec2 = self.dec2(x4,ind4,down4)
        dec3 = self.dec3(x3, ind3) #, down3)
        dec4 = self.dec4(dec3, ind2) #, down2)
        dec5 = self.dec5(dec4, ind1, down1)

        return dec5