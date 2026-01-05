"""
DSH Detector CNN - ResNet Architecture
=======================================
Improved architecture using ResNet with modifications optimized for
sparse X-ray image classification.

Key Changes from Standard CNN:
1. ResNet with skip connections - preserves sharp details
2. LeakyReLU - keeps faint signals from dark pixels alive
3. Strided convolutions - doesn't delete low-intensity halo data
4. Proper initialization for sparse data

Author: DSH Detection Project - Part 5 (Model Selection & Training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    The skip connection allows the network to preserve sharp details
    from earlier layers, critical for distinguishing PSF from halos.
    
    Uses:
    - LeakyReLU instead of ReLU (keeps faint signals alive)
    - Strided convolution for downsampling (doesn't delete low-intensity data)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: nn.Module = None,
        negative_slope: float = 0.01
    ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # LeakyReLU keeps faint signals alive (important for sparse X-ray data)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        
        # Downsample for skip connection if dimensions change
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # This is the key - preserves original information
        out = self.leaky_relu(out)
        
        return out


class DSHResNet(nn.Module):
    """
    ResNet-based DSH Detector optimized for sparse X-ray images.
    
    Architecture designed for:
    - Input: 64x64 grayscale X-ray images
    - Output: Binary classification (halo probability)
    
    Key features:
    - Skip connections preserve sharp PSF vs fuzzy halo distinction
    - LeakyReLU keeps faint halo signals alive
    - Strided convolutions instead of MaxPool (doesn't delete low-intensity data)
    - Dropout for regularization
    """
    
    def __init__(
        self, 
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        negative_slope: float = 0.01
    ):
        super(DSHResNet, self).__init__()
        
        self.negative_slope = negative_slope
        
        # Initial convolution (no pooling!)
        # Input: (1, 64, 64) -> (32, 64, 64)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        
        # Residual layers with strided convolutions for downsampling
        # (32, 64, 64) -> (32, 64, 64)
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        
        # (32, 64, 64) -> (64, 32, 32) - strided conv, no maxpool
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        
        # (64, 32, 32) -> (128, 16, 16)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        
        # (128, 16, 16) -> (256, 8, 8)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # Use strided convolution for downsampling (not MaxPool!)
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # First block may have stride > 1 for downsampling
        layers.append(ResidualBlock(
            in_channels, out_channels, stride, downsample, self.negative_slope
        ))
        
        # Remaining blocks have stride=1
        for _ in range(1, blocks):
            layers.append(ResidualBlock(
                out_channels, out_channels, 1, None, self.negative_slope
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming init for LeakyReLU
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', 
                    nonlinearity='leaky_relu', 
                    a=self.negative_slope
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=self.negative_slope)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, 64, 64)
            
        Returns:
            Probability tensor (batch, 1)
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # Sigmoid for probability
        x = torch.sigmoid(x)
        
        return x
    
    def get_confidence_category(self, probability: float) -> str:
        """Categorize detection confidence."""
        if probability >= 0.85:
            return "DEFINITE_HALO"
        elif probability >= 0.65:
            return "PROBABLE_HALO"
        elif probability >= 0.45:
            return "POSSIBLE_HALO"
        elif probability >= 0.25:
            return "UNLIKELY_HALO"
        else:
            return "NO_HALO"


class DSHResNetLite(nn.Module):
    """
    Lightweight ResNet for faster inference.
    Fewer channels but same architectural improvements.
    """
    
    def __init__(self, dropout_rate: float = 0.2, negative_slope: float = 0.01):
        super(DSHResNetLite, self).__init__()
        
        self.negative_slope = negative_slope
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        
        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128, 1)
        
        self._initialize_weights()
    
    def _make_layer(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride, downsample, self.negative_slope)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='leaky_relu', a=self.negative_slope)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=self.negative_slope)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return torch.sigmoid(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Keep the old model available for comparison
class DSHDetectorCNN(nn.Module):
    """Original VGG-style CNN (kept for backward compatibility)."""
    
    def __init__(self, dropout_rate: float = 0.3):
        super(DSHDetectorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
    def get_confidence_category(self, probability: float) -> str:
        if probability >= 0.85:
            return "DEFINITE_HALO"
        elif probability >= 0.65:
            return "PROBABLE_HALO"
        elif probability >= 0.45:
            return "POSSIBLE_HALO"
        elif probability >= 0.25:
            return "UNLIKELY_HALO"
        else:
            return "NO_HALO"


if __name__ == "__main__":
    print("=" * 60)
    print("DSH Detector Models Comparison")
    print("=" * 60)
    
    # Test ResNet
    resnet = DSHResNet()
    print(f"\nDSHResNet Parameters: {count_parameters(resnet):,}")
    
    # Test ResNet Lite
    resnet_lite = DSHResNetLite()
    print(f"DSHResNetLite Parameters: {count_parameters(resnet_lite):,}")
    
    # Test old CNN
    old_cnn = DSHDetectorCNN()
    print(f"Original CNN Parameters: {count_parameters(old_cnn):,}")
    
    # Test forward pass
    sample = torch.randn(4, 1, 64, 64)
    
    print(f"\nInput shape: {sample.shape}")
    print(f"ResNet output: {resnet(sample).shape}")
    print(f"ResNetLite output: {resnet_lite(sample).shape}")
    print(f"Old CNN output: {old_cnn(sample).shape}")
    
    print("\n" + "=" * 60)
    print("Architecture Improvements:")
    print("=" * 60)
    print("1. Skip connections - preserve sharp PSF vs fuzzy halo details")
    print("2. LeakyReLU - keeps faint signals from dark pixels alive")
    print("3. Strided conv - doesn't delete low-intensity halo data")
    print("4. Proper Kaiming init for LeakyReLU")
    print("=" * 60)