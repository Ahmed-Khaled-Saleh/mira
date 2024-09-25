from rouge import Rouge
import numpy as np
import torch

rouge = Rouge()


def rouge_score(hyp_ids, ref_ids, tokenizer):
    hyps = torch.where(hyp_ids != -100, hyp_ids, tokenizer.pad_token_id)
    hyps = [tokenizer.decode(hyps, skip_special_tokens=True)]
    

    if len(hyps[0]) == 0:
        return 0.0
    
    refs = torch.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)
    refs = [tokenizer.decode(refs, skip_special_tokens=True)]
    
    
    try:
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score


def acc_score(preds, labels):
    preds = np.array(preds.cpu())
    labels = np.array(labels.cpu())
    return np.sum(preds == labels) / float(len(labels))