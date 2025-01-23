from datasets import load_dataset, Dataset
import json
import csv
import openai
import anthropic
import requests
import os
import logging
from tqdm import tqdm
import time
from unsloth.chat_templates import standardize_sharegpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(dataset_source, dataset_path, tokenizer, hf_token=None):
    if dataset_source == 'huggingface':
        try:
            dataset = load_dataset(dataset_path, split="train", use_auth_token=hf_token)
        except ValueError:
            # If use_auth_token is not supported, try without it
            dataset = load_dataset(dataset_path, split="train")
    elif dataset_source == 'local':
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")
        
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            elif isinstance(data, dict):
                dataset = Dataset.from_dict(data)
            else:
                raise ValueError("JSON file must contain either a list or a dictionary.")
        elif dataset_path.endswith('.csv'):
            with open(dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            dataset = Dataset.from_list(data)
        else:
            raise ValueError("Unsupported file format. Please use JSON or CSV.")
    else:
        raise ValueError("Invalid dataset source. Use 'huggingface' or 'local'.")
    print(dataset.column_names)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    print(dataset[5]["conversations"])
    print(dataset[5]["text"])

    return dataset
