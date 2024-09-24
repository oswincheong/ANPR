import os
import numpy as np
import glob
import time
from PIL import Image
import mmap
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import onnxruntime as ort
from transformers import TrOCRProcessor

# Parameters
run = 2

# File path
data_root_base = r'C:\\Users\Oswin\Desktop\FYP\ANPR-1\Datasets\Dataset_all_random'
dataset_type = 'warped'
model_name = 'microsoft/trocr-small-printed'

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT:      str = os.path.join(data_root_base, dataset_type)

@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = model_name

# Initialize ONNX runtime session
onnx_model_path = 'trocr_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Load the processor
processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)

def read_image_mmap(image_path):
    """
    Read an image using memory mapping to optimize file access.
    """
    with open(image_path, "r+b") as f:
        # Memory-map the file, size 0 means the whole file
        mmapped_file = mmap.mmap(f.fileno(), 0)
        # Read the memory-mapped file into a PIL Image
        image = Image.open(mmapped_file)
        image = image.convert('RGB')
        image = image.resize((384, 384))  # Adjust to the size expected by the model
        return image

def single_ocr_onnx(image, processor, ort_session):
    """
    Perform OCR on a single image using ONNX Runtime.
    """
    # Preprocess the image using the processor
    pixel_values = processor(images=[image], return_tensors='np', padding=True).pixel_values

    # Dummy decoder input ids (start token)
    decoder_input_ids = np.ones((1, 1), dtype=np.int64)

    # Prepare the inputs for ONNX
    ort_inputs = {
        "pixel_values": pixel_values,
        "decoder_input_ids": decoder_input_ids
    }

    # Run the ONNX model
    ort_outs = ort_session.run(None, ort_inputs)

    # Get logits and token IDs
    logits = ort_outs[0]  # Assuming first output contains logits
    token_ids = np.argmax(logits, axis=-1)  # Get token IDs by taking the argmax over logits

    # Decode the generated token IDs to text
    generated_text = processor.batch_decode(token_ids, skip_special_tokens=True)

    return logits, token_ids, generated_text[0]

def eval_new_data_onnx(
    data_path='test/*',
    ground_truth_path='test.txt'
):
    total_correct = 0
    total_samples = 0
    total_latency = 0
    generated_texts_all = []  # List to store generated texts
    ground_truth_labels = []  # List to store ground truth labels

    # Read ground truth from file
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = f.readlines()

    # Create a dictionary to store ground truth for quick lookup
    ground_truth_dict = {}
    for line in ground_truth_data:
        filename, ground_truth = line.strip().split('\t')
        ground_truth_dict[filename] = ground_truth

    # Get image paths
    image_paths = glob.glob(data_path)  # Assuming images are in JPEG format

    # Process images one by one
    for image_path in tqdm(image_paths, desc="Processing images"):
        image = read_image_mmap(image_path)
        
        start_time = time.time()
        logits, token_ids, generated_text = single_ocr_onnx(image, processor, ort_session)  # Process single image
        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency

        filename = os.path.basename(image_path)
        ground_truth = ground_truth_dict.get(filename, None)

        print(f"\nImage: {filename}")
        print(f"Logits: {logits}")
        print(f"Token IDs: {token_ids}")
        print(f"Generated text: {generated_text}")
        
        if ground_truth is not None:
            print(f"Ground truth: {ground_truth}")
            # Compare recognized text with ground truth for accuracy measurement
            if generated_text == ground_truth:
                total_correct += 1
            total_samples += 1
            generated_texts_all.append(generated_text)
            ground_truth_labels.append(ground_truth)

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    avg_latency = total_latency / len(image_paths)

    # Calculate precision, recall, and f1 score
    precision = precision_score(ground_truth_labels, generated_texts_all, average='weighted')
    recall = recall_score(ground_truth_labels, generated_texts_all, average='weighted')
    f1 = f1_score(ground_truth_labels, generated_texts_all, average='weighted')

    # Print metrics
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Average latency: {avg_latency:.2f} seconds")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Combine all metrics into a single dictionary
    metrics = {
        'accuracy': accuracy,
        'avg_latency': avg_latency,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics, generated_texts_all, ground_truth_labels

# Run inference with ONNX
metrics, predicted_labels, ground_truth_labels = eval_new_data_onnx(
    data_path='test/*',
    ground_truth_path='test.txt'
)
