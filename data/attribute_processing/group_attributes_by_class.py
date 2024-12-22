#!/usr/bin/env python3
# coding: utf-8

"""
group_attributes_by_class.py

A command-line tool to group attributes by specified class labels using WordNet synsets.

This script processes a JSON file containing word attributes (the output of extract_vg_attributes.py),
identifies words that correspond to the provided class labels based on WordNet synsets, and aggregates
their attributes into a new JSON file organized by class labels.

Usage:
    python group_attributes_by_class.py --input_path <input_json> \
                                        --output_path <output_json> \
                                        --classes <class1> <class2> ...

Example:
    python group_attributes_by_class.py --input_path Object_Attributes.json \
                                        --output_path Grouped_Attributes.json \
                                        --classes airplane automobile bird cat

Dependencies:
    - nltk (specifically WordNet)
    - argparse

Before running, ensure that the NLTK WordNet data is downloaded (e.g., `nltk.download('wordnet')`).

Citation:
    @article{krishna2017visual,
      title={Visual genome: Connecting language and vision using crowdsourced dense image annotations},
      author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and others},
      journal={International journal of computer vision},
      volume={123},
      pages={32--73},
      year={2017},
      publisher={Springer}
}
"""

import json
import argparse
import os
import sys
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Group attributes by specified class labels using WordNet synsets."
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the JSON file produced by extract_vg_attributes.py.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to the output JSON file where grouped attributes will be saved.'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        required=True,
        help='List of class labels to group attributes by.'
    )
    return parser.parse_args()

def download_wordnet():
    try:
        _ = wordnet.synsets('test')
    except LookupError:
        nltk.download('wordnet')

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: Input file '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(path, 'r') as f:
        data = json.load(f)
    data = {k.replace(' ', '_'): v for k, v in data.items()}
    return data

def find_equivalents(class_labels, words):
    equivalents = defaultdict(set)
    for label in class_labels:
        for word in words:
            synsets = wordnet.synsets(word, pos=wordnet.NOUN)
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.name().lower() == label.lower():
                        equivalents[label].add(word)
    return equivalents

def aggregate_attributes(equivalents, data):
    aggregated = defaultdict(list)
    for label, eq_words in equivalents.items():
        for w in eq_words:
            aggregated[label].extend(data.get(w, []))
    aggregated = {k: list(set(v)) for k, v in aggregated.items()}
    return aggregated

def main():
    args = parse_arguments()
    download_wordnet()
    data = load_data(args.input_path)
    words = list(data.keys())
    equivalents = find_equivalents(args.classes, words)
    if not any(equivalents.values()):
        print("Warning: No equivalents found for the provided class labels.")
    grouped = aggregate_attributes(equivalents, data)
    with open(args.output_path, 'w') as f:
        json.dump(grouped, f, indent=4)
    print(f"Grouped attributes saved to '{args.output_path}'.")

if __name__ == "__main__":
    main()
