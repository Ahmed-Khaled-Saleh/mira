
import copy

from enum import Enum
from torch.utils.data import Dataset
import json
import os
from dataclasses import dataclass
import torch
import transformers
import pandas as pd


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




class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


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
        return dict(
            input_ids=input_ids,
            labels=labels,
            categories=categories,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class MTLDataSet(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"], 
                 generation=False):
        """
            list_data_dict: list of dictionaries with keys 'input', 'output', 'category'
        """
        
        super(MTLDataSet, self).__init__()
            
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

import torch
from torch.utils.data import random_split

def split_by_category(list_data_dict):
        groupdict = dict()
        for example in list_data_dict:
            if example['category'] not in groupdict:
                groupdict[example['category']] = []
            groupdict[example['category']].append(example)
        return groupdict


def train_eval_split(dataset):
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

import random

def split_tasks_among_clients(tasks_dict, num_clients=10):
    result = {task: [[] for _ in range(num_clients)] for task in tasks_dict}
    
    for task, samples in tasks_dict.items():
        # Shuffle the samples to ensure randomness
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        # Distribute samples among clients
        for i, sample in enumerate(shuffled_samples):
            client_index = random.randint(0, num_clients - 1)
            result[task][client_index].append(sample)
    
    return result

def get_dolly(args, tokenizer):
    data_collator = LLMDataCollator(tokenizer=tokenizer)
    json_name = 'databricks-dolly-15k.jsonl'
    list_data_dict =  load_jsonl(os.path.join('data', json_name), 
                            instruction='instruction',
                            input='context',
                            output='response',
                            category='category')
    
    grouped_data = split_by_category(list_data_dict)
    result = split_tasks_among_clients(grouped_data)

    lst_train_ds = []
    lst_eval_set = []
    lst_train_ds_genr = []
    lst_eval_set_genr = []

    for task in result:
        for client_data in result[task]:
            dataset = MTLDataSet(client_data, tokenizer, generation=False)
            train_loader, val_loader = train_eval_split(dataset)
            lst_train_ds.append(train_loader)
            lst_eval_set.append(val_loader)

            dataset_genr = MTLDataSet(client_data, tokenizer, generation=True)
            train_ds_genr, val_loader_genr = train_eval_split(dataset_genr)
            lst_train_ds_genr.append(train_ds_genr)
            lst_eval_set_genr.append(val_loader_genr)

    return (lst_train_ds, lst_eval_set, tokenizer, data_collator), (lst_train_ds_genr, lst_eval_set_genr) 
