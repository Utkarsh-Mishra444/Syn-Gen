#!/usr/bin/env python3
# coding: utf-8

"""
generate_gemma_prompts_4bit.py

Generate descriptive prompts using a locally loaded Gemma model in 4-bit quantization.
Similar to the previous gemma script, but:
    - Uses bitsandbytes to load the model in 4-bit precision.
    - Loads the model from a local path instead of a remote Hugging Face model.

Usage Example:
    python generate_gemma_prompts_4bit.py \
        --model_path /path/to/local_gemma_model \
        --quantization 4 \
        --input_json /path/to/prompts_for_SDE.json \
        --output_json Gemma_2B_4bit_Prompts.json \
        --token <YOUR_HF_AUTH_TOKEN> \
        --max_prompts 1000

Dependencies:
    - transformers, torch, bitsandbytes, accelerate, tqdm
    - Make sure your system/GPU supports 4-bit quantization.

Note:
    If you don't need 4-bit quant or local loading, the previous script might suffice.
"""

import argparse
import json
import os
import sys
import time
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Local path to the model directory (containing model weights).'
    )
    parser.add_argument(
        '--token',
        type=str,
        default='',
        help='Optional Hugging Face authentication token if needed.'
    )
    parser.add_argument(
        '--quantization',
        type=int,
        default=4,
        choices=[4, 8],
        help='Load the model in 4-bit or 8-bit quantization via bitsandbytes.'
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
        help='Optional path to also save the output data as a pickle file.'
    )
    parser.add_argument(
        '--max_prompts',
        type=int,
        default=1000,
        help='Max number of prompts to process per class.'
    )
    return parser.parse_args()

def load_quantized_model(model_path, hf_token, quant_bits=4):
    """
    Loads a local model in 4-bit or 8-bit quantization using bitsandbytes.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=(quant_bits == 4),
        load_in_8bit=(quant_bits == 8)
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it", token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quant_config,
        device_map="auto"
    )
    return tokenizer, model

def generate_text_with_instruction(tokenizer, model, prompt_text):
    """
    Given a user prompt, prepend a system instruction and generate
    a visually descriptive text prompt for a text-to-image model.
    """
    system_instruction = (
        "Make a single paragraph very visually descriptive rich text prompt for a text to image model. "
        "The aim is to generate data to train a classifier so realism is the focus. "
        "Use the following sentence as input."
    )
    chat = [
        {"role": "user", "content": system_instruction + prompt_text}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=500)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return full_text.strip()

def main():
    args = parse_arguments()

    # Load local quantized model
    print(f"Loading local model from {args.model_path} with {args.quantization}-bit quantization...")
    tokenizer, model = load_quantized_model(
        model_path=args.model_path,
        hf_token=args.token,
        quant_bits=args.quantization
    )

    # Check input JSON
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    output_data = {}
    total_prompts = 0
    for class_key, prompts in data.items():
        total_prompts += min(args.max_prompts, len(prompts))

    pbar = tqdm(total=total_prompts, desc="Generating Prompts (4-bit)")

    # Process each class
    for class_key, prompts in data.items():
        output_data[class_key] = {}
        prompt_counter = 0

        for prompt_key, prompt_text in prompts.items():
            if prompt_counter >= args.max_prompts:
                break

            generated_text = generate_text_with_instruction(tokenizer, model, prompt_text)
            output_data[class_key][prompt_key] = generated_text

            pbar.update(1)
            prompt_counter += 1

    pbar.close()

    # Save JSON
    with open(args.output_json, 'w') as out_f:
        json.dump(output_data, out_f, indent=4)
    print(f"Prompts generated. Output saved to '{args.output_json}'.")

    # Optional pickle
    if args.pickle_output:
        with open(args.pickle_output, 'wb') as pk_f:
            pickle.dump(output_data, pk_f)
        print(f"Pickle file saved to '{args.pickle_output}'.")

if __name__ == "__main__":
    main()
