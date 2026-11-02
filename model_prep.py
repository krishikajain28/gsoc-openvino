import torch
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

model = models.resnet18( weights = models.ResNet18_Weights.DEFAULT )

print("model loaded succesufully")
