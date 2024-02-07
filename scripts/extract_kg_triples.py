import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.hf_api import HfFolder
import argparse
import re
from collections import defaultdict
import os
import pathlib
import json
from tqdm import tqdm
import xopen


OUTPUT_ROOT = (pathlib.Path(__file__).parent / "output").resolve()
DATA_ROOT = (pathlib.Path(__file__).parent / "data").resolve()


def main(args):

    device = "cuda"

    access_token = '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'
    HfFolder.save_token(access_token)
    name = "meta-llama/Llama-2-7b-chat-hf"

    model = AutoModelForCausalLM.from_pretrained(name, device_map = device,
                                             cache_dir='/scratch/alpine/anra7539').to(device)

    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    with open(DATA_ROOT / args.file) as fin, open(OUTPUT_ROOT / (args.file.split('/')[-1].split('.')[0] + "_triples.json"), "w", encoding='utf-8') as fout:

        data = json.load(fin)
        data = data.iloc[6150:,:]

        # For each row
        for i in tqdm(range(len(data['question']))):
            triples = []

            docs_to_extract = []
            docs_to_extract.extend([doc for doc in data['gold_docs'][str(i)]['paragraph']])
            docs_to_extract.extend([doc for doc in data['distractor_docs'][str(i)]['paragraph']])

            for doc in docs_to_extract:
                triples.append(generate_triples(doc, model, tokenizer, device))

            fout.write(json.dumps(triples) + "\n")


def generate_triples(text, model, tokenizer, device):

    triples = []
    for chunk in text:
        prompt = f"Extract knowledge graph triples of the form (subject, relation, object) from the following text (the generated text should only include triples and nothing else): {chunk}\nKnowledge graph triples:"
        input_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

        outputs = model.generate(**input_tokens)

        triples.append(
            "\n".join(tokenizer.decode(outputs[0], skip_special_tokens=True).split("Knowledge graph triples:")[1:]))

    return triples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check for duplicate sentences in a text file.')

    parser.add_argument('file', metavar='file', type=str, help='Path to the input file.')

    #parser.add_argument(
    #    "--model", "-m",
    #    required=True,
    #    help='Huggingface model string to use for KG triple extraction.'
    #)

    parser.add_argument("--batch-size", help="Batch size use in generation", type=int, default=4)

    args = parser.parse_args()
    main(args)