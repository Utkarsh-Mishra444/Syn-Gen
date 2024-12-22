#!/usr/bin/env python3
# coding: utf-8

"""
generate_confusion_matrices.py

Evaluate one or more PyTorch model checkpoints on the CIFAR-10 test set, then generate
and save confusion matrices (as PNG files) for each model.

Features:
    - Load standard ResNet architectures (resnet18, resnet34, etc.) or a custom model.
    - Process all .pth files in a specified folder.
    - Print accuracy and save confusion matrices as "<model_file>_confusion_matrix.png".

Usage Examples:

1) Standard ResNet34:
   python generate_confusion_matrices.py \
       --model_folder /path/to/resnet_checkpoints \
       --model_name resnet34 \
       --data_path ./data \
       --batch_size 4 \
       --output_dir ./confusion_matrices

2) Custom Model:
   python generate_confusion_matrices.py \
       --model_folder /path/to/custom_checkpoints \
       --model_name custom \
       --custom_model_module models.my_resnet_impl \
       --custom_model_class MyResNet \
       --data_path ./data \
       --batch_size 4 \
       --output_dir ./confusion_matrices

Dependencies:
    - torch, torchvision, matplotlib, numpy
"""

import os
import argparse
import numpy as np
import torch
import torchvision
import importlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152
)
from torch.utils.data import DataLoader

CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_folder', type=str, required=True,
        help='Path to the folder containing .pth model checkpoints.'
    )
    parser.add_argument(
        '--model_name', type=str, default='resnet34',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'custom'],
        help='ResNet architecture to load or "custom" for a user-defined model.'
    )
    parser.add_argument(
        '--custom_model_module', type=str, default=None,
        help='Python module path for custom model (e.g., "models.my_resnet"). Required if model_name=custom.'
    )
    parser.add_argument(
        '--custom_model_class', type=str, default=None,
        help='Class name for custom model (e.g., "MyResNet"). Required if model_name=custom.'
    )
    parser.add_argument(
        '--data_path', type=str, default='./data',
        help='Path to the CIFAR-10 data directory. Downloaded if not found.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for the CIFAR-10 test loader.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Directory where confusion matrix PNG files are saved.'
    )
    return parser.parse_args()

def get_standard_resnet(model_name):
    if model_name == 'resnet18':
        return resnet18(pretrained=False)
    elif model_name == 'resnet34':
        return resnet34(pretrained=False)
    elif model_name == 'resnet50':
        return resnet50(pretrained=False)
    elif model_name == 'resnet101':
        return resnet101(pretrained=False)
    elif model_name == 'resnet152':
        return resnet152(pretrained=False)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

def load_custom_model(module_path, class_name):
    if not module_path or not class_name:
        raise ValueError("Both --custom_model_module and --custom_model_class must be provided for a custom model.")
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"Cannot find class '{class_name}' in module '{module_path}'.")
    return model_class()

def create_confusion_matrix_image(cm, classes, model_file, output_dir):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(model_file)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    filename = f"{os.path.splitext(model_file)[0]}_confusion_matrix.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CIFAR-10 test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Prepare base model (standard or custom)
    def initialize_model():
        if args.model_name == 'custom':
            return load_custom_model(args.custom_model_module, args.custom_model_class).to(device)
        else:
            return get_standard_resnet(args.model_name).to(device)

    # Evaluate each .pth file in model_folder
    for model_file in os.listdir(args.model_folder):
        if model_file.endswith('.pth'):
            model_path = os.path.join(args.model_folder, model_file)

            # Load a fresh model for each checkpoint
            model = initialize_model()
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            confusion_matrix = np.zeros((10, 10), dtype=int)
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    for i in range(len(labels)):
                        confusion_matrix[labels[i]][predicted[i]] += 1

            accuracy = 100.0 * correct / total
            print(f"Model: {model_file}, Accuracy: {accuracy:.2f}%")

            # Create and save confusion matrix
            create_confusion_matrix_image(confusion_matrix, CIFAR10_CLASSES, model_file, args.output_dir)

if __name__ == "__main__":
    main()
