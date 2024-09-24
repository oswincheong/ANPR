import onnx

# Load the ONNX model
onnx_model_path = 'trocr_model.onnx'  # Replace with your ONNX model path
model = onnx.load(onnx_model_path)

# Iterate through the initializers (which hold the weights)
for initializer in model.graph.initializer:
    weight_name = initializer.name
    weight_data = onnx.numpy_helper.to_array(initializer)
    
    print(f"Weight: {weight_name} | Shape: {weight_data.shape}")
    print(weight_data)  # This will print the actual weight values
