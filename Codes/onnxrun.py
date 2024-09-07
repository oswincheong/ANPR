import torch
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from PIL import Image

# Define the device (CPU in this case)
device = torch.device('cpu')

# Initialize the model and processor using Optimum's ORTModelForSeq2SeqLM
model_name = "checkpoint-3300"  # Replace with your actual model checkpoint
model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')

# Example image input (replace with your actual image path)
image_path = r'C:\Users\Oswin\Desktop\FYP\ANPR-1\Datasets\Dataset_all_random\warped\test\warped_00081_VL3734.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB

# Preprocess the image
pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)

# Prepare decoder input ids (usually starting with the BOS token)
bos_token_id = processor.tokenizer.bos_token_id
decoder_input_ids = torch.tensor([[bos_token_id]], device=device)

# Run inference with the ONNX model
with torch.no_grad():
    ort_outs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids)

# Decode the token IDs to text
decoded_text = processor.batch_decode(ort_outs, skip_special_tokens=True)[0]
print("Decoded text:", decoded_text)
