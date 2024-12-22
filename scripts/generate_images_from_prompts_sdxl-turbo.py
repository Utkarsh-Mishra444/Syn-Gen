#!/usr/bin/env python3
# coding: utf-8

"""
generate_images_from_prompts_sdxl-turbo.py

Generate images from class-based prompts using a Diffusers text-to-image pipeline,
optionally resizing images to 32x32. Outputs are saved locally, and can optionally
be zipped into an archive.

Features:
    - Loads prompts from a JSON file structured like:
        {
            "airplane": {
                "0": "Prompt text",
                "1": "Another prompt text",
                ...
            },
            "automobile": {
                ...
            }
            ...
        }
    - For each class, generates a specified number of images from the stored prompts.
    - Uses the SDXL Turbo model ("stabilityai/sdxl-turbo") by default.
    - Optionally resizes images to 32x32 for CIFAR-like usage (default behavior).
    - Can zip the entire output directory.

Usage Example:
    python generate_images_from_prompts.py \
        --prompts_json Gemma_2B_Instruct_Generated_Prompts.json \
        --output_dir cifar_synthetic_images \
        --num_images_per_class 1000 \
        --zip_output \
        --no_resize  # if you don't want 32x32 resizing

Dependencies:
    - diffusers, transformers, accelerate, torch, Pillow (PIL), tqdm
"""

import os
import json
import shutil
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers import AutoPipelineForText2Image

# Default CIFAR-10 classes
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompts_json', type=str, required=True,
        help='Path to the JSON file containing prompts. '
             'Should be structured like {class_name: {index: "prompt_text", ...}, ...}.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='cifar_synthetic_images',
        help='Directory to save generated images.'
    )
    parser.add_argument(
        '--num_images_per_class', type=int, default=1000,
        help='How many images to generate per class.'
    )
    parser.add_argument(
        '--model_id', type=str, default='stabilityai/sdxl-turbo',
        help='Hugging Face model ID or local path for the Diffusers pipeline.'
    )
    parser.add_argument(
        '--inference_steps', type=int, default=1,
        help='Number of inference (denoising) steps for generation.'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=0.0,
        help='Guidance scale for classifier-free guidance.'
    )
    parser.add_argument(
        '--classes', nargs='*', default=None,
        help='List of classes to generate. By default, uses all CIFAR-10 classes.'
    )
    parser.add_argument(
        '--zip_output', action='store_true',
        help='If set, zip the output directory after generation.'
    )
    parser.add_argument(
        '--no_resize', action='store_true',
        help='If set, do not resize images to 32x32. By default, images are resized.'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load prompts JSON
    if not os.path.exists(args.prompts_json):
        raise FileNotFoundError(f"Prompts JSON file not found: {args.prompts_json}")
    with open(args.prompts_json, 'r') as f:
        prompts_dict = json.load(f)

    # Determine classes
    class_labels = args.classes if args.classes else CIFAR10_CLASSES

    # Initialize pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    total_images = args.num_images_per_class * len(class_labels)
    pbar = tqdm(total=total_images, desc="Generating images")
    image_count = 0

    for class_name in class_labels:
        # Make subfolder for each class
        class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Retrieve prompts for this class
        if class_name not in prompts_dict:
            print(f"Warning: Class '{class_name}' not found in JSON. Skipping.")
            continue

        prompt_items = list(prompts_dict[class_name].items())

        # Generate up to num_images_per_class
        for i in range(args.num_images_per_class):
            if i >= len(prompt_items):
                # Not enough prompts in the JSON for this class
                break

            _, prompt_text = prompt_items[i]
            # Generate image
            result = pipe(
                prompt=prompt_text,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale
            )
            image = result.images[0]

            # Optionally resize to 32x32
            if not args.no_resize:
                image = image.resize((32, 32), Image.Resampling.LANCZOS)

            # Save
            filename = f"{class_name}_{i+1}.png"
            image.save(os.path.join(class_dir, filename))

            image_count += 1
            pbar.update(1)

    pbar.close()
    print(f"Finished generating {image_count} images in '{args.output_dir}'.")

    # Optionally zip
    if args.zip_output:
        archive_path = shutil.make_archive(args.output_dir, 'zip', args.output_dir)
        print(f"Output directory zipped to: {archive_path}")

if __name__ == "__main__":
    main()
