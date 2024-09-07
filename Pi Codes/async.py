import os
import mmap  # Import mmap
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import asyncio
from collections import defaultdict
from PIL import Image
from dataclasses import dataclass
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.asyncio import tqdm_asyncio  # Use tqdm for asyncio

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

# Load the pre-trained model without quantization
trained_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-' + str(3300)).to(device)

async def read_image_mmap(image_path):
    """
    Asynchronously read an image using memory mapping to optimize file access.
    """
    with open(image_path, "r+b") as f:
        # Memory-map the file, size 0 means whole file
        mmapped_file = mmap.mmap(f.fileno(), 0)
        # Read the memory-mapped file into a PIL Image
        image = Image.open(mmapped_file)
        image = image.convert('RGB')
        image = image.resize((224, 224))  # Adjust to a smaller size
        return image

async def ocr_async(image, processor, model):
    """
    Asynchronously perform OCR on an image.
    """
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]

async def process_image(image_path, processor, model):
    """
    Asynchronously load and process a single image.
    """
    image = await read_image_mmap(image_path)
    text = await ocr_async(image, processor, model)
    return image_path, text

async def eval_new_data(
    data_path=r'/home/oswin/Documents/ANPR/Dataset/test',
    ground_truth_path=r'\home\oswin\Documents\ANPR\Dataset\test.txt',
    batch_size=4  # Adjust batch size based on memory constraints
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

    # Prepare tasks for processing images asynchronously
    tasks = [process_image(image_path, processor, trained_model) for image_path in image_paths]

    # Process images asynchronously with progress tracking
    for task in tqdm_asyncio.as_completed(tasks, desc="Processing images", total=len(tasks)):
        filename, text = await task
        start_time = time.time()
        
        ground_truth = ground_truth_dict.get(os.path.basename(filename), None)
        if ground_truth is not None:
            # Compare recognized text with ground truth for accuracy measurement
            if text == ground_truth:
                total_correct += 1
            total_samples += 1
            ground_truth_labels.append(ground_truth)

        generated_texts.append(text)
        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency

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

# Run inference asynchronously
metrics, predicted_labels, ground_truth_labels = asyncio.run(eval_new_data(
    data_path='test/*',
    ground_truth_path='test.txt'
))
