import torch
from torch.utils.data import Dataset

class WMT14Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len, src_lang, tgt_lang, pad_token="<pad>"):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Truncate or pad to max_len
        self.tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token, length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset['translation'][index]
        src_text = example[self.src_lang]
        tgt_text = example[self.tgt_lang]

        src_enc = self.tokenizer.encode(src_text)
        tgt_enc = self.tokenizer.encode(tgt_text)

        src_ids = torch.tensor(src_enc.ids, dtype=torch.long)
        src_mask = torch.tensor(src_enc.attention_mask, dtype=torch.long)

        tgt_ids = torch.tensor(tgt_enc.ids, dtype=torch.long)
        tgt_mask = torch.tensor(tgt_enc.attention_mask, dtype=torch.long)

        return {
            'src_ids': src_ids,
            'src_mask': src_mask,
            'tgt_ids': tgt_ids,
            'tgt_mask': tgt_mask
        }