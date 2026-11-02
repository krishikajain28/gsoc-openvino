import numpy as np
import openvino as ov

core = ov.Core()
model = core.read_model("resnet18_ir.xml")
compiled_model = core.compile_model(model=model, device_name="CPU")
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
output_layer = compiled_model.output(0)
result = compiled_model(dummy_input)[output_layer]

print(f"Inference complete. Output shape: {result.shape}")