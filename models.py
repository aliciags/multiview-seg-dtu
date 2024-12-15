import torch
import torch.nn as nn
import torch.nn.functional as F
# import segmentation_models_pytorch as smp

class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        
        # Downsampling (Encoder)
        self.conv1 = nn.Conv2d(11, 32, kernel_size=3, padding=1)  # Input channels: 11
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Upsampling (Decoder)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)  # Output channels: 1 for binary mask
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Downsample by factor of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Downsample by factor of 2
        
        # Decoder
        x = F.relu(self.upconv1(x))
        x = torch.sigmoid(self.upconv2(x))  # Sigmoid to keep output in range [0, 1]
        
        return x


# improvements: batch normalization, dropout for regularization, and skip connections for better spatial feature retention
class SegmentationModel(nn.Module):
    def __init__(self):  # Correctly use __init__ here
        super(SegmentationModel, self).__init__()  # Correctly use __init__ here
        
        # Encoder: Downsampling Path
        self.enc1 = self._conv_block(11, 64)    # Input channels: 11 (BF images)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder: Upsampling Path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # Skip connection from enc3
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # Skip connection from enc2
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)   # Skip connection from enc1
        
        # Output Layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)  # Output channels: 1 for binary mask
    
    def _conv_block(self, in_channels, out_channels):
        """
        A helper function to create a convolutional block with Conv2d, BatchNorm, and ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = F.max_pool2d(enc1, 2)  # Downsample by factor of 2
        
        enc2 = self.enc2(x)
        x = F.max_pool2d(enc2, 2)
        
        enc3 = self.enc3(x)
        x = F.max_pool2d(enc3, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv3(x)
        x = torch.cat((x, enc3), dim=1)  # Concatenate with skip connection
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec1(x)
        
        # Output Layer
        x = torch.sigmoid(self.out_conv(x))  # Sigmoid for binary mask
        
        return x


def pretrained_UNet():
    # Load pretrained U-Net with a ResNet backbone
    pretrained_unet = smp.Unet(
        encoder_name="resnet34",        # Choose a ResNet backbone (others available)
        encoder_weights="imagenet",    # Use pretrained weights on ImageNet
        in_channels=3,                 # Placeholder for in_channels, will replace later
        classes=1                      # Output channels (binary mask)
    )
    
    # Modify the first convolutional layer to accept 11 input channels
    pretrained_unet.encoder.conv1 = nn.Conv2d(
        in_channels=11,               # From 3 to 11 channels
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    # Optional: Initialize weights for the modified layer
    nn.init.kaiming_normal_(pretrained_unet.encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    return pretrained_unet
