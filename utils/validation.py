from rouge import Rouge
import numpy as np
import torch

rouge = Rouge()


def rouge_score(hyp_ids, ref_ids, tokenizer):
    hyps = torch.where(hyp_ids != -100, hyp_ids, tokenizer.pad_token_id)
    refs = torch.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)

    hyps = tokenizer.batch_decode(hyps, skip_special_tokens=True)
    refs = tokenizer.batch_decode(refs, skip_special_tokens=True)
    
    batch_rouge = 0
    for i in range(len(hyps)):
        if len(hyps[i].strip()) == 0:
            continue
        
        else:
            h = hyps[i].strip().lower()
            r = refs[i].strip().lower()
            try:
                item_rouge = rouge.get_scores(h, r)[0]['rouge-l']['f']
            except ValueError:
                print("Error in calculating rouge score")
                item_rouge = 0

            batch_rouge += item_rouge

    rouge_score = batch_rouge / len(hyps)
    
    return rouge_score


def acc_score(preds, labels):
    preds = np.array(preds.cpu())
    labels = np.array(labels.cpu())
    return np.sum(preds == labels) / float(len(labels))