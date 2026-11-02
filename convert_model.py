import openvino as ov
from torchvision.models import resnet18, ResNet18_Weights
import torch

model = resnet18(weights=ResNet18_Weights.DEFAULT)

dummy_input = torch.randn(1, 3, 224, 224)

ov_model = ov.convert_model(model, example_input=dummy_input)

ov.save_model(ov_model, "resnet18_ir.xml")
