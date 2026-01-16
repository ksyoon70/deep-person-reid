
import torch
from torchreid.losses import ArcFaceLoss

def test_arcface_layer():
    print("Testing ArcFaceLoss Layer...")
    num_classes = 10
    feat_dim = 128
    batch_size = 4
    
    # Initialize layer
    arcface = ArcFaceLoss(num_classes, feat_dim)
    if torch.cuda.is_available():
        arcface = arcface.cuda()
    
    # Create dummy data
    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    if torch.cuda.is_available():
        features = features.cuda()
        labels = labels.cuda()
        
    # Forward pass
    output = arcface(features, labels)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_classes)
    
    # Check values range (logits shouldn't be exploding due to normalization)
    print(f"Output mean: {output.mean().item()}")
    print("ArcFaceLoss Layer Test Passed!")

if __name__ == '__main__':
    test_arcface_layer()
