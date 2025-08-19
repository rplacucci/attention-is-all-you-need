import torch
from torch.utils.data import Dataset

class WMT14Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset['translation'][index]
        
        return {
            k: torch.tensor(self.tokenizer.encode(v).ids, dtype=torch.long)
            for k, v in example.items()
        }