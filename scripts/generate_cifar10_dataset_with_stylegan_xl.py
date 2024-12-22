#!/usr/bin/env python3
# coding: utf-8

"""
generate_cifar10_dataset_with_stylegan_xl.py

Generate a full CIFAR-10-like dataset using StyleGAN-XL's CIFAR-10 model. This script:
  - Loops over specified class indices (default 0..9).
  - For each class, calls generate_images.py with a given range of seeds
    (for example, 0..(images_per_class-1)).
  - Saves the resulting images in subfolders (one per class).

Requirements:
    - Must run inside (or point to) the StyleGAN-XL repo directory,
      where generate_images.py is located.
    - The 'cifar10.pkl' model should be downloaded or placed locally.
    - Python, PyTorch, etc., as required by StyleGAN-XL.

Example Usage:
    1) Clone the repo & install:
       git clone https://github.com/autonomousvision/stylegan_xl.git
       cd stylegan_xl
       pip install -r requirements.txt
    2) Download CIFAR-10 model:
       wget https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl
    3) Run this script (while in stylegan_xl folder):
       python ../generate_cifar10_dataset_with_stylegan_xl.py \
           --model_path /path/to/cifar10.pkl \
           --outdir cifar10_generated \
           --images_per_class 5000 \
           --classes 0 1 2 3 4 5 6 7 8 9

That would produce 50k images (5k per each of 10 classes), with each class in its own subfolder.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the StyleGAN-XL CIFAR-10 pickle file (e.g., cifar10.pkl).'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='cifar10_generated',
        help='Base output directory for generated images (subfolders per class).'
    )
    parser.add_argument(
        '--images_per_class',
        type=int,
        default=1000,
        help='How many images to generate for each class.'
    )
    parser.add_argument(
        '--batch_sz',
        type=int,
        default=1,
        help='Batch size per sample (passed to generate_images.py).'
    )
    parser.add_argument(
        '--classes',
        nargs='*',
        type=int,
        default=None,
        help='List of class indices to generate. If not specified, uses all 0..9.'
    )
    parser.add_argument(
        '--truncation_psi',
        type=float,
        default=1.0,
        help='Truncation psi (passed to generate_images.py).'
    )
    parser.add_argument(
        '--noise_mode',
        type=str,
        default='const',
        choices=['const', 'random', 'none'],
        help='Noise mode for generate_images.py.'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Confirm generate_images.py is present
    script_path = os.path.join('.', 'generate_images.py')
    if not os.path.isfile(script_path):
        print("[ERROR] 'generate_images.py' not found. Make sure this script is in or points to the stylegan_xl directory.")
        sys.exit(1)

    # Default to CIFAR-10 classes 0..9 if none specified
    if args.classes is None or len(args.classes) == 0:
        class_indices = list(range(10))  # 0..9 for CIFAR-10
    else:
        class_indices = args.classes

    # Create base outdir
    os.makedirs(args.outdir, exist_ok=True)

    # For each class, call generate_images.py with the required seeds
    for cls_idx in class_indices:
        class_subdir = os.path.join(args.outdir, f'class_{cls_idx}')
        os.makedirs(class_subdir, exist_ok=True)

        # Example: seeds=0..(images_per_class - 1)
        seed_range_str = f"0-{args.images_per_class - 1}"

        cmd = [
            sys.executable,                   # python executable
            script_path,
            f'--network={args.model_path}',   # model path
            f'--seeds={seed_range_str}',      # e.g. "0-4999" if images_per_class=5000
            f'--batch-sz={args.batch_sz}',
            f'--trunc={args.truncation_psi}',
            f'--noise-mode={args.noise_mode}',
            f'--outdir={class_subdir}',
            f'--class={cls_idx}'  # important to generate images for this class
        ]

        print("\n======================================================")
        print(f"Generating class {cls_idx}: {args.images_per_class} images")
        print("Command:", " ".join(cmd))
        print("======================================================\n")

        subprocess.run(cmd, check=True)

    print("\nAll specified classes have been generated successfully!\n")
    print(f"Your datasets are located in {args.outdir}")

if __name__ == "__main__":
    main()
