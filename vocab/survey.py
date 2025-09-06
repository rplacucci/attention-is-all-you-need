import os
import sys
import argparse
import numpy as np

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer

# Config argparser
parser = argparse.ArgumentParser(description="Train the original Transformer for language translation")
parser.add_argument("--lang", type=str, default="de", help="Langauge to translate to/from English ('cs', 'de', 'fr', 'hi', 'ru)")
args = parser.parse_args()

# Setup tokenizer
lang_a = args.lang
lang_b = "en"
lang_pair = f"{lang_a}-{lang_b}"

path_vocab = f"./vocab/wmt14_{lang_pair}/bpe_{lang_pair}.json"
if not os.path.exists(path_vocab):
    print(f"Tokenizer file {path_vocab} does not exist. Run 'python -m vocab.build_vocab' and try again")
    sys.exit(1)

tokenizer = Tokenizer.from_file(path_vocab)
vocab_size = tokenizer.get_vocab_size()
print(f"Loaded tokenizer from {path_vocab} with size {vocab_size:,}")

# Load dataset
max_len = 512
wmt14 = load_dataset("wmt/wmt14", lang_pair, split="train")

def add_lengths(batch):
    src = [item["de"] for item in batch["translation"]]
    tgt = [item["en"] for item in batch["translation"]]
    src_ids = [tokenizer.encode(s).ids for s in src]
    tgt_ids = [tokenizer.encode(t).ids for t in tgt]
    return {
        "src_len": [len(x) for x in src_ids],
        "tgt_len": [len(x) for x in tgt_ids],
    }

# Save src and tgt token lengths
path_cache = f"./vocab/wmt14_{lang_pair}/train_len_cache"

ds = wmt14.map(add_lengths, batched=True, num_proc=4)
ds.select_columns(["src_len", "tgt_len"]).flatten_indices().save_to_disk(path_cache)

# Check lengths
ds_len = Dataset.load_from_disk(path_cache)
train_lengths = np.stack([ds_len["src_len"], ds_len["tgt_len"]], axis=1).astype("int32")
print(f"Loaded train_lengths from {path_cache} with shape {train_lengths.shape}")