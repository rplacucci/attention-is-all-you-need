import torch
import torch.nn as nn
    
def make_src_allow_mask(src_ids, pad_id):
    # [B, S] -> [B, 1, 1, S], 1=token, 0=pad
    return (src_ids != pad_id).unsqueeze(1).unsqueeze(2)

def make_tgt_allow_mask(tgt_ids, pad_id):
    # build padding allow mask [B, 1, 1, T]
    pad_allow = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)  # 1=token, 0=pad
    # build causal allow mask [1, T, T] lower triangle of ones
    T = tgt_ids.size(1)
    causal = torch.tril(torch.ones((T, T), device=tgt_ids.device, dtype=torch.bool)).unsqueeze(0)
    # combine -> [B, 1, T, T], still 1=allow, 0=block
    return pad_allow & causal