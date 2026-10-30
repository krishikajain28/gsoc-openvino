import openvino as ov
import numpy as np
import cv2
from torchvision.models import ResNet18_Weights

core = ov.Core()
model = core.read_model("resnet18_ir.xml")
compiled_model = core.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

labels = ResNet18_Weights.DEFAULT.meta["categories"]

image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("Image not found!")

print("Loaded image shape:", image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))
image = np.transpose(image, (2,0,1))
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

# normalizing the image
image = image / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
image = (image - mean) / std

result = compiled_model(image)[output_layer]
print("Result :" , result.shape)

predicted_index = np.argmax(result)
print("Predicted Class Index:", predicted_index)

predicted_object = labels[predicted_index]
print("Predicted Object: ", predicted_object)

