import random
from re import split
from datasets import load_dataset

DATA_PATH = {
    "alpaca": "/home/pairshoe/.cache/huggingface/hub/datasets--tatsu-lab--alpaca/snapshots/dce01c9b08f87459cf36a430d809084718273017",
    "gsm8k": "/home/pairshoe/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/e53f048856ff4f594e959d75785d2c2d37b678ee",
    "squad": "/home/pairshoe/.cache/huggingface/hub/datasets--rajpurkar--squad/snapshots/7b6d24c440a36b6815f21b70d25016731768db1f",
    "swag": "/home/pairshoe/.cache/huggingface/hub/datasets--allenai--swag/snapshots/dc48148372b3853a9c7bae7bb06c161b46d8364a"
}

def load_dataset_sample(dataset_name, sample_size):
    if 'alpaca' in dataset_name:    
        """加载数据集并随机采样"""
        ds = load_dataset(DATA_PATH['alpaca'])
        all_texts = []
        random_numbers = random.sample(range(len(ds['train']['instruction'])), sample_size)
        for i in random_numbers:
            all_texts.append(ds['train']['instruction'][i] + (ds['train']['input'][i] or ""))
        return all_texts
    elif 'gsm8k' in dataset_name:
        ds = load_dataset(DATA_PATH['gsm8k'],name='main',split='test')
        all_texts = []
        random_numbers = random.sample(range(len(ds)), sample_size)
        for i in random_numbers:
            all_texts.append(ds[i]['question'])
        return all_texts
    elif 'squad' in dataset_name:
        ds = load_dataset(DATA_PATH['squad'],split="validation")
        all_texts = []
        random_numbers = random.sample(range(len(ds)), sample_size)
        for i in random_numbers:
            all_texts.append(ds[i]['context']+(ds[i]['question']))
        return all_texts
    elif 'swag' in dataset_name:
        ds = load_dataset(DATA_PATH['swag'],split='test')
        all_texts = []
        random_numbers = random.sample(range(len(ds)), sample_size)
        for i in random_numbers:
            all_texts.append(ds[i]['startphrase'])
        return all_texts
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")