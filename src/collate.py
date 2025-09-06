import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch, pad_id):
    # Collect tokens
    src_ids = [b["src_ids"] for b in batch]
    tgt_ids = [b["tgt_ids"] for b in batch]
    src_mask = [b["src_mask"] for b in batch]
    tgt_mask = [b["tgt_mask"] for b in batch]

    # Pad to max length found *in this batch*
    src_ids = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
    src_mask = pad_sequence(src_mask, batch_first=True, padding_value=0)
    tgt_mask = pad_sequence(tgt_mask, batch_first=True, padding_value=0)

    return {
        'src_ids': src_ids,
        'src_mask': src_mask,
        'tgt_ids': tgt_ids,
        'tgt_mask': tgt_mask
    }