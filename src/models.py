import torch
import torchvision.models as models

def load_resnet50(device="cuda"):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    m.eval()
    return m
