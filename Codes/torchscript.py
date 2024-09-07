import os
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from PIL import Image
from dataclasses import dataclass
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Parameters
batch_size = 8
epochs = 20
learning_rate = 0.0001
run = 2

# File path
data_root_base = r'C:\\Users\Oswin\Desktop\FYP\ANPR-1\Datasets\Dataset_all_random'
dataset_type = 'warped'
model_name = 'microsoft/trocr-small-printed'

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:    int = batch_size
    EPOCHS:        int = epochs
    LEARNING_RATE: float = learning_rate

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT:      str = os.path.join(data_root_base, dataset_type)

@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = model_name

device = torch.device('cpu')
processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)

# Load the pre-trained model
model = VisionEncoderDecoderModel.from_pretrained('checkpoint-'+str(3300)).to(device)

# Convert the model to TorchScript using JIT
model.eval()  # Switch to evaluation mode
scripted_model = torch.jit.script(model)  # Script the model
# Optional: Save the scripted model for later use
# scripted_model.save("scripted_trocr_model.pt")

def read_and_preprocess_image(image_path):
    """
    Read and preprocess image: resizing it to a smaller size.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Adjust to a smaller size
    return image

def ocr(image, processor, model):
    """
    Perform OCR on a single image.
    """
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]

def eval_new_data(
    data_path=r'/home/oswin/Documents/ANPR/Dataset/test',
    ground_truth_path=r'\home\oswin\Documents\ANPR\Dataset\test.txt'
):
    total_correct = 0
    total_samples = 0
    total_latency = 0
    generated_texts = []  # List to store generated texts
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

    for image_path in tqdm(image_paths, desc="Processing images"):
        image = read_and_preprocess_image(image_path)
        start_time = time.time()
        text = ocr(image, processor, scripted_model)  # Use scripted model
        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency

        generated_texts.append(text)
        filename = os.path.basename(image_path)
        ground_truth = ground_truth_dict.get(filename, None)

        if ground_truth is not None:
            # Compare recognized text with ground truth for accuracy measurement
            if text == ground_truth:
                total_correct += 1
            total_samples += 1
            ground_truth_labels.append(ground_truth)

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    avg_latency = total_latency / len(image_paths)

    # Calculate precision, recall, and f1 score
    precision = precision_score(ground_truth_labels, generated_texts, average='weighted')
    recall = recall_score(ground_truth_labels, generated_texts, average='weighted')
    f1 = f1_score(ground_truth_labels, generated_texts, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}%")
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

    return metrics, generated_texts, ground_truth_labels

# Run inference
metrics, predicted_labels, ground_truth_labels = eval_new_data(
    data_path='test/*',
    ground_truth_path='test.txt'
)
