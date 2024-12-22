#!/usr/bin/env python3
# coding: utf-8

"""
train_mixed_datasets.py

Train a ResNet (or custom model) on a mixture of real (CIFAR-10) data and synthetic data.

Features:
    - Select a standard ResNet architecture from torchvision (resnet18, resnet34, resnet50, resnet101, resnet152)
      or load a custom model from a user-supplied Python module.
    - Mix real and synthetic data according to a ratio from 0 to 1 in configurable increments.
    - Evaluate on CIFAR-10 test set.
    - Log training metrics (loss, accuracy) to a text file.
    - Use mixed-precision training with torch.cuda.amp.

Usage Example:
    python train_mixed_datasets.py \
        --real_dataset_path /path/to/cifar10 \
        --synthetic_dataset_path /path/to/synthetic_data \
        --model_name resnet34 \
        --batch_size 1024 \
        --epochs 30 \
        --ratio_increments 6 \
        --log_file training_metrics.txt

Or load a custom model class:
    python train_mixed_datasets.py \
        --real_dataset_path /path/to/cifar10 \
        --synthetic_dataset_path /path/to/synthetic_data \
        --model_name custom \
        --custom_model_module models.my_resnet_impl \
        --custom_model_class MyResNet \
        --batch_size 1024 \
        --epochs 30

Notes:
    - Ensure you have installed all necessary libraries (torch, torchvision).
    - CIFAR-10 is automatically downloaded if not present.
    - Synthetic data should follow an ImageFolder structure (class subfolders).
    - If using a custom model, it must be importable, and the class should match --custom_model_class.
"""

import os
import sys
import argparse
import importlib
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms, datasets
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--real_dataset_path', type=str, required=True,
        help='Path to the CIFAR-10 data directory (downloaded if not found).'
    )
    parser.add_argument(
        '--synthetic_dataset_path', type=str, required=True,
        help='Path to the synthetic data directory (ImageFolder structure).'
    )
    parser.add_argument(
        '--model_name', type=str, default='resnet34',
        help='Which model to train. One of (resnet18, resnet34, resnet50, resnet101, resnet152) or "custom".'
    )
    parser.add_argument(
        '--custom_model_module', type=str, default=None,
        help='If model_name="custom", specify the import path for your custom model module (e.g., "models.my_resnet_impl").'
    )
    parser.add_argument(
        '--custom_model_class', type=str, default=None,
        help='If model_name="custom", specify the class name for your custom model (e.g., "MyResNet").'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='Batch size for DataLoader.'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Number of epochs for training.'
    )
    parser.add_argument(
        '--log_file', type=str, default='training_metrics.txt',
        help='File to log training metrics.'
    )
    parser.add_argument(
        '--ratio_increments', type=int, default=6,
        help='Number of ratio increments (0 to 1.0 in steps of 1/(ratio_increments-1)).'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate.'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='Weight decay for optimizer.'
    )
    parser.add_argument(
        '--scheduler_step', type=int, default=20,
        help='Step size for learning rate scheduler.'
    )
    parser.add_argument(
        '--scheduler_gamma', type=float, default=0.1,
        help='Gamma for learning rate scheduler.'
    )
    return parser.parse_args()

def create_mixed_dataset(real_dataset, synthetic_dataset, synthetic_ratio, base_dataset_size=None):
    """
    Creates a mixed dataset of real and synthetic data based on a ratio.

    Args:
        real_dataset:       A PyTorch dataset (e.g., CIFAR-10).
        synthetic_dataset:  A PyTorch dataset (ImageFolder).
        synthetic_ratio:    Fraction of total data to be synthetic.
        base_dataset_size:  Total number of samples in the mixed dataset.
                            If None, defaults to min(len(real_dataset), len(synthetic_dataset)).

    Returns:
        ConcatDataset of real and synthetic subsets.
    """
    if base_dataset_size is None:
        base_dataset_size = min(len(real_dataset), len(synthetic_dataset))

    n_synthetic = int(base_dataset_size * synthetic_ratio)
    n_real = base_dataset_size - n_synthetic

    synthetic_indices = torch.randperm(len(synthetic_dataset))[:n_synthetic]
    real_indices = torch.randperm(len(real_dataset))[:n_real]

    synthetic_subset = Subset(synthetic_dataset, synthetic_indices)
    real_subset = Subset(real_dataset, real_indices)

    mixed_dataset = ConcatDataset([synthetic_subset, real_subset])
    return mixed_dataset

def get_standard_resnet(model_name):
    """
    Returns an untrained standard ResNet model (no pretraining).
    """
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
        raise ValueError(f"Unsupported model_name: {model_name}")

def load_custom_model(module_path, class_name):
    """
    Dynamically import a custom model module and instantiate its class.

    Args:
        module_path: e.g. "models.my_resnet_impl"
        class_name: e.g. "MyResNet"

    Returns:
        An instance of the specified custom model class.
    """
    if module_path is None or class_name is None:
        raise ValueError("For a custom model, both --custom_model_module and --custom_model_class are required.")

    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"Cannot find class '{class_name}' in module '{module_path}'.")
    return model_class()

def main():
    args = parse_arguments()

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure images match CIFAR-10 dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Real dataset (CIFAR-10)
    real_dataset = datasets.CIFAR10(
        root=args.real_dataset_path,
        train=True,
        download=True,
        transform=transform
    )
    # Test set
    testset = datasets.CIFAR10(
        root=args.real_dataset_path,
        train=False,
        download=True,
        transform=transform
    )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Synthetic dataset
    synthetic_dataset = datasets.ImageFolder(root=args.synthetic_dataset_path, transform=transform)

    # Logging
    with open(args.log_file, "w") as file:
        file.write("Ratio, Epoch, Loss, Accuracy\n")

        # Generate ratio values from 0.0 to 1.0
        step_size = 1.0 / max(1, (args.ratio_increments - 1))
        ratio_values = [round(i * step_size, 2) for i in range(args.ratio_increments)]

        for ratio in ratio_values:
            print(f"\n===== Training with ratio: {ratio} =====")

            # Create model (standard or custom)
            if args.model_name.lower() == "custom":
                model = load_custom_model(args.custom_model_module, args.custom_model_class).to(device)
            else:
                model = get_standard_resnet(args.model_name).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
            scaler = amp.GradScaler()

            # Create mixed dataset & loader
            mixed_dataset = create_mixed_dataset(real_dataset, synthetic_dataset, ratio)
            train_loader = DataLoader(
                mixed_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2
            )

            # Train
            for epoch in range(args.epochs):
                model.train()
                running_loss = 0.0

                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)

                # Evaluate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100.0 * correct / total

                # Log
                file.write(f"{ratio:.2f}, {epoch+1}, {epoch_loss:.6f}, {accuracy:.2f}%\n")
                file.flush()

                print(f"Ratio {ratio:.2f}, Epoch {epoch+1}, Loss {epoch_loss:.6f}, Accuracy {accuracy:.2f}%")

                scheduler.step()

            # Save model
            model_filename = f"{args.model_name}_ratio{int(ratio*100)}_acc{accuracy:.2f}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved: {model_filename}")

if __name__ == "__main__":
    main()
