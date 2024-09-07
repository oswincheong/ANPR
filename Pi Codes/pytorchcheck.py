import torch
import onnxruntime as ort
import numpy as np
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image

# Define the device (CPU in this case)
device = torch.device('cpu')

# Initialize the model and processor
torch_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-3300').to(device)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')

# Function to convert PyTorch tensor to NumPy array
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Example image input (make sure to replace with the actual path to your image)
image_path = '/home/oswin/Documents/ANPR/Codes/test/warped_00081_VL3734.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB

# Preprocess the image
dummy_input = processor(image, return_tensors='pt').pixel_values.to(device)

# Prepare decoder input ids (usually starting with the BOS token)
bos_token_id = processor.tokenizer.bos_token_id
decoder_input_ids = torch.tensor([[bos_token_id]], device=device)

# 1. Run the model in PyTorch
torch_model.eval()  # Set the model to evaluation mode

# Get PyTorch output (assuming this model generates sequences)
torch_out = torch_model.generate(dummy_input, decoder_input_ids=decoder_input_ids)

# Convert PyTorch output to NumPy
torch_out_numpy = to_numpy(torch_out)

# 2. Run the model in ONNX Runtime
# Load the ONNX model
ort_session = ort.InferenceSession("trocr_model.onnx")

# Prepare the inputs for ONNX Runtime
ort_inputs = {
    ort_session.get_inputs()[0].name: to_numpy(dummy_input),
    ort_session.get_inputs()[1].name: to_numpy(decoder_input_ids)
}

# Get ONNX Runtime output (raw logits or a single token)
ort_outs = ort_session.run(None, ort_inputs)

# If ONNX output is a single value, compare just the first token
ort_out_token = np.argmax(ort_outs[0], axis=-1) if ort_outs[0].ndim > 2 else ort_outs[0]

# Compare only the first token
try:
    np.testing.assert_array_equal(torch_out_numpy[:, 0], ort_out_token)
    print("First token from PyTorch and ONNX Runtime outputs match.")
except AssertionError as e:
    print("First token from PyTorch and ONNX Runtime outputs do not match!")
    print(str(e))
