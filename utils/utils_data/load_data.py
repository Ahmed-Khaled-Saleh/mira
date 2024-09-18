import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.utils_data.partition_data import partition_idx_labeldir
from collections import Counter
from utils.utils_data.subset import get_tokenizer, subset

def get_datasets(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    
    tokenizer = get_tokenizer(args)

    # Generation task
    # if args.dataset == 'dolly':
    #     from utils.utils_data.load_dolly import LLMDataset, LLMDataCollator
        
    #     if args.eval_metric == 'loss':
    #         raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=False)
    #     else:
    #         raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=True)

    #     data_collator = LLMDataCollator(tokenizer=tokenizer)

    #     # only use a subset of raw dataset
    #     train_set, eval_set, y_train = subset(args, raw_datasets)

    #     counter = Counter(y_train)
    #     noniid = args.iid

    #     if 'dir' in noniid:
    #         split_dic = partition_idx_labeldir(y_train, n_parties=args.num_clients, alpha=float(noniid[3:]), num_classes=len(counter))
    #         split_trainsets = []
    #         for _, sample_indices in split_dic.items():
    #             split_trainsets.append(Subset(train_set, indices=sample_indices))
    #     else:
    #         n_parts = [int(len(train_set) / args.num_clients) for _ in range(args.num_clients - 1)]
    #         n_parts.append(len(train_set) - sum(n_parts))
    #         split_trainsets = torch.utils.data.dataset.random_split(train_set, n_parts)

    #     list_train_loader = [
    #         DataLoader(
    #             subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    #         ) for subset in split_trainsets
    #     ]
    #     eval_loader = DataLoader(
    #         eval_set, batch_size=args.batch_size, collate_fn=data_collator
    #     )
        

    if args.dataset == 'dolly':
        from utils.utils_data.dolly_mtl import get_dolly
        (lst_train_ds, lst_eval_set, tokenizer, data_collator), (lst_train_ds_genr, lst_eval_set_genr) = get_dolly(args, tokenizer)
        return (lst_train_ds, lst_eval_set, tokenizer, data_collator), (lst_train_ds_genr, lst_eval_set_genr)
    elif args.dataset in ['instruct']:
        from utils.utils_data.ni_loader import get_instruction_dataset
        (lst_train_ds, lst_eval_set, tokenizer, data_collator), (lst_train_ds_genr, lst_eval_set_genr) = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
        return (lst_train_ds, lst_eval_set, tokenizer, data_collator), (lst_train_ds_genr, lst_eval_set_genr)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
