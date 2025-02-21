from typing import Optional
from enum import Enum

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from collections import defaultdict
import random

MAX_LENGTH = 640


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def read_data(data_path: str) -> list[str]:
    with open(f"{data_path}/train-00000-of-00002.txt", 'r') as f:
        data = f.readlines()
    
    with open(f"{data_path}/train-00001-of-00002.txt", 'r') as f:
        data.extend(f.readlines())

    num_samples = 100000
    data = data[:num_samples]
    
    return data


def tokenize_data(data: list[str], tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH) -> list[torch.Tensor]:
    if max_length is None:
        return [tokenizer(text)["input_ids"] for text in data]
    else:
        return [tokenizer(text)["input_ids"][:max_length] for text in data]


def pad_tensor(input_ids, max_length: int, pad_token: int) -> torch.Tensor:
    if len(input_ids) < max_length:
        return input_ids + [pad_token] * (max_length - len(input_ids))
    else:
        return input_ids[:max_length]


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, tokenizer: AutoTokenizer = None):
        self.data_path = data_path
        self.tokenizer = tokenizer

        self.data = read_data(data_path)
        self.tokenized_data = tokenize_data(self.data, tokenizer, max_length)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.tokenized_data[idx]


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, tokenizer: AutoTokenizer = None):
        self.data_path = data_path
        self.tokenizer = tokenizer

        self.data = read_data(data_path)
        self.tokenized_data = tokenize_data(self.data, tokenizer, max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.tokenized_data[idx]


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, tokenizer: AutoTokenizer = None):
        self.data = read_data(data_path)
        self.tokenizer = tokenizer
        self.tokenized_data = tokenize_data(self.data, tokenizer, max_length=None)
        self.lengths = [len(text) for text in self.tokenized_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, tokenizer: AutoTokenizer = None):
        self.data = read_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = tokenize_data(self.data, tokenizer, max_length)
        random.shuffle(self.tokenized_data)
        self._metadata()
        random.shuffle(self.metadata)

    def __len__(self):
        return len(self.metadata)
    
    def block_diagonal_mask(self, segments: list[tuple[int, int]]) -> torch.Tensor:
        # print(segments)
        full_length = segments[-1][1]
        mask = torch.zeros(full_length, full_length, dtype=torch.float)
        for start, end in segments:
            mask[start:end, start:end] = torch.tril(torch.ones(end - start, end - start))  # Block diagonal causal mask
        return mask
    
    def _metadata(self,):
        self.metadata = []

        offset = 0
        current_seq = []
        segments_for_current_seq = []

        for tokenized_text in self.tokenized_data:
            end = offset + len(tokenized_text)

            if end >= self.max_length:
                while offset + len(tokenized_text) > self.max_length:
                    current_seq.extend(tokenized_text[:self.max_length - offset])
                    segments_for_current_seq.append((offset, self.max_length))
                    tokenized_text = tokenized_text[self.max_length - offset:]
                    
                    self.metadata.append({
                        "seq": current_seq,
                        "segments": segments_for_current_seq
                    })
                    
                    current_seq = []
                    segments_for_current_seq = []
                    offset = 0
            else:
                current_seq.extend(tokenized_text)
                segments_for_current_seq.append((offset, end))
                offset = end
    
    def __getitem__(self, idx: int):
        metadata = self.metadata[idx]
        input_ids = torch.tensor(metadata["seq"])
        attention_mask = self.block_diagonal_mask(metadata["segments"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        } 


def collate_fn(
    input_ids: list[torch.Tensor], 
    max_length: Optional[int] = MAX_LENGTH, 
    tokenizer: AutoTokenizer = None,
    data_mode: DataMode = DataMode.BRAIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    if data_mode == DataMode.BRAIN:
        input_ids = [pad_tensor(t, max_length, tokenizer.pad_token_id) for t in input_ids]
        input_ids = torch.tensor(input_ids)
    elif data_mode == DataMode.BIG_BRAIN:
        max_length = min(max_length, max([len(t) for t in input_ids]))
        input_ids = [pad_tensor(t, max_length, tokenizer.pad_token_id) for t in input_ids]
        input_ids = torch.tensor(input_ids)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        max_length = min(max_length, max([len(t) for t in input_ids]))
        input_ids = [pad_tensor(t, max_length, tokenizer.pad_token_id) for t in input_ids]
        input_ids = torch.tensor(input_ids)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        pass
    return input_ids


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, k):
        """
        Args:
            dataset (UltraBigBrainDataset): The dataset to sample from.
            batch_size (int): The number of samples per batch.
            k (int): Maximum allowed difference between longest and shortest sample in a batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        
        # Group samples by length
        self.length_buckets = defaultdict(list)
        for idx, length in enumerate(dataset.lengths):
            self.length_buckets[length].append(idx)
        
        self.sorted_lengths = sorted(self.length_buckets.keys())
        self.batches = [torch.tensor(batch) for batch in self._create_batches()]
    
    def _create_batches(self):
        batches = []
        
        # Create sorted groups of indices within k-length constraints
        length_groups = []
        temp_group = []
        min_len = None
        
        for length in self.sorted_lengths:
            indices = self.length_buckets[length]
            
            if not temp_group:
                min_len = length
            
            if length - min_len > self.k:
                num_full_batches = len(temp_group) // self.batch_size
                if num_full_batches > 0:
                    length_groups.append(temp_group[:num_full_batches * self.batch_size])
                    temp_group = temp_group[num_full_batches * self.batch_size:] + indices[:]
                    min_len = len(self.dataset.tokenized_data[temp_group[0]])
                else:
                    new_min_len = length - self.k
                    temp_group = [t for t in temp_group if len(self.dataset.tokenized_data[t]) >= new_min_len] + indices[:]
                    min_len = new_min_len
            else:
                temp_group.extend(indices)
        
        if temp_group:
            num_full_batches = len(temp_group) // self.batch_size
            if num_full_batches > 0:
                length_groups.append(temp_group[:num_full_batches * self.batch_size])
        
        # Create batches
        for group in length_groups:
            random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batches.append(group[i:i + self.batch_size])
        
        for batch in batches:
            assert len(batch) == self.batch_size
        
        random.shuffle(batches)
        return batches
    
    def __iter__(self):
        yield from self.batches
    
    def __len__(self):
        return len(self.batches)
