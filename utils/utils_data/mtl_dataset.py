"""
The implementation of loading dataset `Dolly-15K` is adapted from [FederatedScope](https://github.com/alibaba/FederatedScope/tree/llm)
"""
import copy

from enum import Enum
from torch.utils.data import Dataset
import json
import os
from dataclasses import dataclass
import torch
import transformers
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from data.utils_data.default_tokens import DefaultToken
from data.utils_data.partition_data import partition_idx_labeldir
from collections import Counter
import os

def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category'):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict



TASKS = {'open_qa': 3742,
         'general_qa': 2191,
         'classification': 2136,
         'closed_qa': 1773,
         'brainstorming': 1766,
         'information_extraction': 1506,
         'summarization': 1188,
         'creative_writing': 709}

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response for the task request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response for the task request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


class LLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"], generation=False):
        super(LLMDataset, self).__init__()
        if dataset == 'dolly':
            json_name = 'databricks-dolly-15k.jsonl'
            list_data_dict =  load_jsonl(os.path.join('data', json_name), 
                                        instruction='instruction',
                                        input='context',
                                        output='response',
                                        category='category')
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, targets, tokenizer, generation=generation)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        self.tasks = categories
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer, generation):
        if generation:
            sources_tokenized, labels_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (sources, targets)
            ]
            input_ids = self._tokenize_fn(sources, tokenizer)["input_ids"]
            labels = self._tokenize_fn(targets, tokenizer)["input_ids"]
        else:
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (examples, sources)
            ]
            input_ids = examples_tokenized["input_ids"]
            labels = copy.deepcopy(input_ids)
            for label, source_len in zip(labels,
                                        sources_tokenized["input_ids_lens"]):
                label[:source_len] = DefaultToken.IGNORE_INDEX.value
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i],
                    tasks=self.tasks[i])



class MTLDataset(Dataset):
    def __init__(self, ids, labels, categories, tasks):
        self.input_ids = ids
        self.labels = labels
        self.categories = categories
        self.tasks = tasks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
            labels=self.labels[i],
            categories=self.categories[i],
            tasks=self.tasks[i])
        
    
def filter_dataset_by_task(dataset, task_key):
    ids = []
    labels = []
    tasks = []
    categories = []
    
    for i in range(len(dataset)):
        input_id, label, category, task = dataset[i].values()
        if task == task_key:
            ids.append(input_id)
            labels.append(label)
            categories.append(category)
            tasks.append(task)
    return MTLDataset(ids, labels, categories, tasks)
    
@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        categories = torch.tensor([instance["categories"] for instance in instances])
        tasks = [instance["tasks"] for instance in instances]
        return dict(
            input_ids=input_ids,
            labels=labels,
            categories=categories,
            tasks=tasks,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    


def get_loaders(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)

    # Generation task
    if args.dataset == 'dolly':
        if args.eval_metric == 'loss':
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=False)
        else:
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=True)

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        list_train_loader = []
        list_eval_loader = []
        for task in TASKS.keys():
            dataset = filter_dataset_by_task(raw_datasets, task)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, [train_size, test_size])
            
            n_parts = [int(len(train_dataset) / args.num_clients_per_task) for _ in range(args.num_clients_per_task - 1)]
            n_parts.append(len(train_dataset) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_dataset, n_parts)

            n_parts_test = [int(len(test_dataset) / args.num_clients_per_task) for _ in range(args.num_clients_per_task - 1)]
            n_parts_test.append(len(test_dataset) - sum(n_parts_test))
            split_testsets = torch.utils.data.dataset.random_split(test_dataset, n_parts_test)

            for subset in split_trainsets:
                list_train_loader.append(
                    DataLoader(
                        subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
                ))
            for subset in split_testsets:
                list_eval_loader.append(
                    DataLoader(
                        subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
                ))

        
    elif args.dataset in ['instruct']:
        from data.utils_data.natural_instruction_loader import get_instruction_dataset
        list_train_loader, eval_loader = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
    return list_train_loader, list_eval_loader, tokenizer
