#!/usr/bin/env python3
# coding: utf-8

"""
generate_gemma_prompts.py

Generate detailed text prompts using the Gemma 2B model (or similar) given a JSON of input phrases.
Outputs are saved in a new JSON file, optionally pickle as well.

Usage Example:
    python generate_gemma_prompts.py \
        --input_json prompts_for_SDE.json \
        --output_json Gemma_2B_Instruct_Generated_Prompts.json \
        --model_id google/gemma-1.1-2b-it \
        --token <YOUR_HF_AUTH_TOKEN> \
        --max_prompts 1000

Dependencies:
    - transformers, torch, tqdm
    - accelerate (if using device_map="auto")
"""

import argparse
import json
import os
import sys
import time
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='google/gemma-1.1-2b-it',
        help='Hugging Face model identifier (e.g., "google/gemma-1.1-2b-it").'
    )
    parser.add_argument(
        '--token',
        type=str,
        default='',
        help='Optional Hugging Face authentication token if required.'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to the JSON file with input prompts.'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='Gemma_2B_Instruct_Generated_Prompts.json',
        help='Output JSON file to save generated prompts.'
    )
    parser.add_argument(
        '--pickle_output',
        type=str,
        default='',
        help='Optional path to save the output data as a pickle file.'
    )
    parser.add_argument(
        '--max_prompts',
        type=int,
        default=1000,
        help='Max number of prompts to process per class. (Set large if you want all.)'
    )
    parser.add_argument(
        '--device_map',
        type=str,
        default='auto',
        help='Device map for model loading (e.g., "auto").'
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='float16',
        help='Torch dtype for model (e.g., "float16", "bfloat16", "float32").'
    )
    parser.add_argument(
        '--revision',
        type=str,
        default='float16',
        help='Model revision branch to use (e.g., "float16").'
    )
    return parser.parse_args()

def generate_text_with_instruction(tokenizer, model, prompt_text):
    """
    Given a user prompt, prepend a system instruction and generate
    a visually descriptive text prompt for a text-to-image model.
    """
    system_instruction = (
        "Make a single paragraph very visually descriptive rich text prompt for a text to image model. "
        "The aim is to generate data to train a classifier so realism is the focus."
        "Use the following sentence as input."
    )
    chat = [
        {"role": "user", "content": system_instruction + prompt_text}
    ]
    # Convert chat to model input
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=500)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to trim extraneous text if needed
    # (Your original snippet uses substring logic to extract from model output,
    # but it depends on certain tokens in Gemma's output. Adjust if you prefer.)
    return full_text.strip()

def main():
    args = parse_arguments()

    # Load model and tokenizer
    print(f"Loading tokenizer from '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=args.token
    )

    print(f"Loading model from '{args.model_id}'...")
    torch_dtype = getattr(torch, args.torch_dtype) if hasattr(torch, args.torch_dtype) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        revision=args.revision,
        token=args.token
    )

    # Load input JSON
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # Prepare output structure
    output_data = {}

    # Count total number of prompts (for TQDM)
    total_prompts = 0
    for class_key, prompts in data.items():
        total_prompts += min(args.max_prompts, len(prompts))

    pbar = tqdm(total=total_prompts, desc="Generating Prompts")

    # Process each class in the JSON
    for class_key, prompts in data.items():
        output_data[class_key] = {}
        prompt_counter = 0

        # For each prompt in that class
        for prompt_key, prompt_text in prompts.items():
            if prompt_counter >= args.max_prompts:
                break

            generated_text = generate_text_with_instruction(tokenizer, model, prompt_text)
            output_data[class_key][prompt_key] = generated_text

            pbar.update(1)
            prompt_counter += 1

    pbar.close()

    # Save the generated outputs into a new JSON file
    with open(args.output_json, 'w') as out_f:
        json.dump(output_data, out_f, indent=4)
    print(f"Prompts generated. Output saved to '{args.output_json}'.")

    # Optionally save as pickle
    if args.pickle_output:
        with open(args.pickle_output, 'wb') as pk_f:
            pickle.dump(output_data, pk_f)
        print(f"Pickle file saved to '{args.pickle_output}'.")

if __name__ == "__main__":
    main()
