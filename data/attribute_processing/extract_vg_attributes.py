#!/usr/bin/env python3
# coding: utf-8

"""
extract_vg_attributes.py

Command-line tool to parse Visual Genome attributes and produce a JSON mapping from 'name' to 'attributes'.

Usage:
    python extract_vg_attributes.py --input_path <input_json> --output_path <output_json>

Example:
    python extract_vg_attributes.py --input_path attributes.json --output_path Object_Attributes.json
"""

import json
import argparse
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract a mapping of 'name' -> 'attributes' from Visual Genome attributes data."
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the Visual Genome attributes JSON file (e.g., attributes.json).'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to the output JSON file where the mapping will be saved (e.g., Object_Attributes.json).'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.exists(args.input_path):
        print(f"Error: The file '{args.input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(args.input_path, 'r') as f:
        data = json.load(f)

    out = {}

    # Original logic
    for i in range(len(data)):
        for j in range(len(data[i]["attributes"])):
            if 'attributes' in data[i]["attributes"][j]:
                a = data[i]["attributes"][j]["names"]
                b = data[i]["attributes"][j]["attributes"]
                #c = data[i]["attributes"][j]["synsets"]  
                for name in a:
                    if name not in out:
                        out[name] = []
                    for attr in b:
                        if attr not in out[name]:
                            out[name].append(attr)

    with open(args.output_path, 'w') as f:
        json.dump(out, f, indent=4)

if __name__ == "__main__":
    main()
