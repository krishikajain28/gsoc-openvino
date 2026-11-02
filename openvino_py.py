import openvino as ov

core = ov.Core()

print("Available Devices: ", core.available_devices)