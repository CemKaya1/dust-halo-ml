"""
Dust Scattering Halo (DSH) Detection CNN
=========================================
Custom CNN architecture for detecting dust scattering halos in X-ray images.
Designed for 64x64 pixel input images from synthetic simulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSHDetectorCNN(nn.Module):
    """
    Custom CNN for Dust Scattering Halo detection.
    
    Architecture designed for:
    - Input: 64x64 grayscale X-ray images
    - Output: Binary classification with confidence score
    - Features: Batch normalization, dropout for regularization
    
    The network uses progressively smaller feature maps with increasing
    channel depth to capture both fine-grained halo structures and 
    global morphological patterns.
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        super(DSHDetectorCNN, self).__init__()
        
        # Convolutional Block 1: 64x64 -> 32x32
        # Captures low-level features like edges and intensity gradients
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 2: 32x32 -> 16x16
        # Captures mid-level features like ring structures
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 3: 16x16 -> 8x8
        # Captures high-level features like overall halo morphology
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 4: 8x8 -> 4x4
        # Captures global patterns and context
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        # Reduces 4x4x256 to 1x1x256, making it spatially invariant
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer (single neuron for binary classification)
        self.fc3 = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activations."""
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
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64)
            
        Returns:
            Tensor of shape (batch_size, 1) with sigmoid probabilities
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output with sigmoid for probability
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
    def get_confidence_category(self, probability: float) -> str:
        """
        Categorize detection confidence based on probability.
        
        Args:
            probability: Model output probability [0, 1]
            
        Returns:
            String category for the detection confidence
        """
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


class DSHDetectorCNNLite(nn.Module):
    """
    Lightweight version of DSH Detector for faster inference.
    Suitable for sliding window detection on large survey images.
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super(DSHDetectorCNNLite, self).__init__()
        
        # Simplified architecture with fewer parameters
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("DSH Detector CNN - Model Summary")
    print("=" * 60)
    
    # Test full model
    model = DSHDetectorCNN()
    print(f"\nFull Model Parameters: {count_parameters(model):,}")
    
    # Test with sample input
    sample_input = torch.randn(4, 1, 64, 64)
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output.squeeze().detach().numpy()}")
    
    # Test confidence categories
    print("\nConfidence Categories:")
    for prob in [0.95, 0.75, 0.55, 0.35, 0.15]:
        cat = model.get_confidence_category(prob)
        print(f"  Probability {prob:.2f} -> {cat}")
    
    print("\n" + "=" * 60)
    
    # Test lite model
    model_lite = DSHDetectorCNNLite()
    print(f"\nLite Model Parameters: {count_parameters(model_lite):,}")
    output_lite = model_lite(sample_input)
    print(f"Lite Output shape: {output_lite.shape}")
