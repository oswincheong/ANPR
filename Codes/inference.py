import os
import torch
import evaluate
import numpy as np
import pandas as pd
import glob as glob
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import csv
import time
import string

from collections import defaultdict
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from urllib.request import urlretrieve
from transformers import (
    VisionEncoderDecoderModel, 
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from optimum import ORTModelForSeq2SeqLM


block_plot = False
plt.rcParams['figure.figsize'] = (12, 9)

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
trained_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-'+str(3300)).to(device)

def read_and_show(image_path):
    """
    :param image_path: String, path to the input image.

    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image

def ocr(image, processor, model):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.

    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def eval_new_data(
    data_path=r'/home/oswin/Documents/ANPR/Dataset/test',
    ground_truth_path=r'\home\oswin\Documents\ANPR\Dataset\test.txt',
):
    total_correct = 0
    total_samples = 0
    total_latency = 0
    generated_texts = []  # List to store generated texts
    ground_truth_labels = [] # List to store ground truth labels
    
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

   # Calculate number of rows and columns for subplots
    ncols = 4  # Adjust based on your preference
    nrows = (len(image_paths) + ncols - 1) // ncols
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows))
    
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        if i == len(image_paths):
            break
        
        # Read and process image
        image = read_and_show(image_path)
        # Measure latency
        start_time = time.time()
        text = ocr(image, processor, trained_model)
        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency
        
        # Store generated text
        generated_texts.append(text)
        
        # Get ground truth from filename
        filename = os.path.basename(image_path)
        ground_truth = ground_truth_dict.get(filename, None)
        
        if ground_truth is not None:
            # Compare recognized text with ground truth for accuracy measurement
            if text == ground_truth:
                total_correct += 1
            total_samples += 1

            # Store ground truth label
            ground_truth_labels.append(ground_truth)
        
        # # Display image and recognized text
        # plt.figure(figsize=(7, 4))
        # plt.imshow(image)
        # plt.title(text)
        # plt.axis('off')
        # plt.show()
        # Plot image and recognized text
        row = i // ncols
        col = i % ncols
        axs[row, col].imshow(image)
        axs[row, col].set_title(text)
        axs[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(image_paths), nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis('off')
    
    # Calculate accuracy
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    # Calculate average latency
    avg_latency = total_latency / len(image_paths) 
    
    # Calculate precision, recall, and f1 score
    precision = precision_score(ground_truth_labels, generated_texts, average='weighted')
    recall = recall_score(ground_truth_labels, generated_texts, average='weighted')
    f1 = f1_score(ground_truth_labels, generated_texts, average='weighted')

    # Print accuracy, latency, precision, recall, and f1 score
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
    
    plt.tight_layout()
    plot_path = os.path.join('/home/oswin/Documents/ANPR', "inference.png")
    plt.savefig(plot_path)
    
    return metrics, generated_texts, ground_truth_labels

metrics, predicted_labels, ground_truth_labels = eval_new_data(
    data_path='test/*',
    ground_truth_path='test.txt')