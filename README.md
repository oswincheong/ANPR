# Automatic Number Plate Recognition (ANPR) using TROCR

This repository contains code for Automatic Number Plate Recognition (ANPR) tasks, primarily focusing on using Tr-OCR (Transformer-based Optical Character Recognition) for optical character recognition.

## Introduction

ANPR is a technology that uses optical character recognition on images to read vehicle registration plates. This repository provides a solution for ANPR tasks, leveraging Tr-OCR.

## Links
- [Tr-OCR Documentation](https://huggingface.co/docs/transformers/v4.40.0/en/model_doc/trocr#overview) 
- [Fine-Tuning Tr-OCR video](https://www.youtube.com/watch?v=-8a7j6EVjs0)

## Features

- **Tr-OCR Integration**: Utilizes Tr-OCR for robust optical character recognition.
- **ANPR Functionality**: Provides functionality for extracting and recognizing vehicle registration plates from images.
- **Fine-tuning**: Fine-tune Tr-OCR on our own Malaysian Carplate dataset for improved performance.
- **Easy-to-Use**: Simple and straightforward implementation for ANPR tasks.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/oswincheong/ANPR.git

## Usage steps

1. Get the label file by using **check_labels_image_name.py**, or **generate_labels.py**
2. Sort the labels in the alphabetical order using **sort_labels.py**
3. Split the dataset using **split_occurences.py**
4. Delete the excess image files based on the upper bound using **delete_excess.py**
5. Configure the file path and parameters in **main.ipynb** and run.
