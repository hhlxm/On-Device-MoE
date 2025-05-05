import random
from re import split

import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset

DATA_PATH = {
    "alpaca": "/home/fit/renju/WORK/lxm/datasets/alpaca",
    "gsm8k": "/home/fit/renju/WORK/lxm/datasets/gsm8k",
    "squad": "/home/fit/renju/WORK/lxm/datasets/squad",
    "swag": "/home/fit/renju/WORK/lxm/datasets/swag",
    "humaneval": "/home/fit/renju/WORK/lxm/datasets/humaneval",
    "alpaca-zh": "/home/fit/renju/WORK/lxm/datasets/alpaca-zh",
    "wikitext-2-raw-v1": "/home/fit/renju/WORK/lxm/datasets/wikitext/wikitext-2-raw-v1"
}

def load_dataset_sample(dataset_name, sample_size):
    if 'alpaca' in dataset_name:    
        """加载数据集并随机采样"""
        ds = load_dataset(DATA_PATH[f'{dataset_name}'])
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
    elif 'humaneval' in dataset_name:
        ds = load_dataset(DATA_PATH['humaneval'],split='test')
        all_texts = []
        random_numbers = random.sample(range(len(ds)), sample_size)
        for i in random_numbers:
            all_texts.append(ds[i]['prompt'])
        return all_texts
    elif 'wikitext-2-raw-v1' in dataset_name:
        ds = load_dataset(DATA_PATH['wikitext-2-raw-v1'],split='test')
        all_texts = []
        random_numbers = random.sample(range(len(ds)), sample_size)
        for i in random_numbers:
            all_texts.append(ds[i]['text'])
        return all_texts
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, data_type, tokenizer, max_length=256):
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据集
        self.dataset = None
        self._load_data()
        
    def _load_data(self):
        """加载整个数据集"""
        if 'alpaca' == self.dataset_name:
            assert self.data_type in ['train'], "data_type must be one of ['train']"
            self.dataset = load_dataset(DATA_PATH[self.dataset_name])[f'{self.data_type}']
            
        elif 'alpaca-zh' == self.dataset_name:
            assert self.data_type in ['train'], "data_type must be one of ['train']"
            self.dataset = load_dataset(DATA_PATH[self.dataset_name])[f'{self.data_type}']
            
        
        elif 'gsm8k' in self.dataset_name:
            assert self.data_type in ['train', 'test'], "data_type must be one of ['train', 'test']"
            self.dataset = load_dataset(DATA_PATH['gsm8k'], name='main', split=f'{self.data_type}')
            
        
        elif 'squad' in self.dataset_name:
            assert self.data_type in ['train', 'validation'], "data_type must be one of ['train', 'validation']"
            self.dataset = load_dataset(DATA_PATH['squad'], split=f'{self.data_type}')
            
        
        elif 'swag' in self.dataset_name:
            assert self.data_type in ['train', 'validation', 'test'], "data_type must be one of ['train', 'validation', 'test']"
            self.dataset = load_dataset(DATA_PATH['swag'], split=f'{self.data_type}')
            
        
        elif 'humaneval' in self.dataset_name:
            assert self.data_type in ['test'], "data_type must be one of ['test']"
            self.dataset = load_dataset(DATA_PATH['humaneval'], split=f'{self.data_type}')
            
        
        elif 'wikitext-2-raw-v1' in self.dataset_name:
            assert self.data_type in ['train', 'validation', 'test'], "data_type must be one of ['train', 'validation', 'test']"
            self.dataset = load_dataset(DATA_PATH['wikitext-2-raw-v1'], split=f'{self.data_type}')
            
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_name == 'alpaca' or self.dataset_name == 'alpaca-zh':
            text = self.dataset['instruction'][idx] + (self.dataset['input'][idx] or "")
        elif 'gsm8k' in self.dataset_name:
            text = self.dataset[idx]['question']
        elif 'squad' in self.dataset_name:
            text = self.dataset[idx]['context'] + self.dataset[idx]['question']
        elif 'swag' in self.dataset_name:
            text = self.dataset[idx]['startphrase']
        elif 'humaneval' in self.dataset_name:
            text = self.dataset[idx]['prompt']
        elif 'wikitext-2-raw-v1' in self.dataset_name:
            text = self.dataset[idx]['text']
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt',padding_side='left'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # 忽略 <pad>
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

   