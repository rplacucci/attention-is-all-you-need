import os
import sys
import argparse
import numpy as np

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer

# Config argparser
parser = argparse.ArgumentParser(description="Train the original Transformer for language translation")
parser.add_argument("--src_lang", type=str, default="de", help="Source language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--tgt_lang", type=str, default="de", help="Target language ('cs', 'de', 'fr', 'hi', 'ru)")
args = parser.parse_args()

# Setup tokenizer
src_lang = args.src_lang
tgt_lang = args.tgt_lang
lang_pair = f"{src_lang}-{tgt_lang}"

path_vocab = f"./vocab/wmt14_{lang_pair}/bpe_{lang_pair}.json"
if not os.path.exists(path_vocab):
    print(f"Tokenizer file {path_vocab} does not exist. Run 'python -m vocab.build_vocab' and try again")
    sys.exit(1)

tokenizer = Tokenizer.from_file(path_vocab)
vocab_size = tokenizer.get_vocab_size()
print(f"Loaded tokenizer from {path_vocab} with size {vocab_size:,}")

# Load dataset
max_len = 512
try:
    wmt14 = load_dataset("wmt/wmt14", lang_pair, split="train")
    dataset_lang_pair = lang_pair
except:
    # If the pair doesn't exist, try reversing it (e.g., "en-de" -> "de-en")
    reversed_pair = f"{tgt_lang}-{src_lang}"
    try:
        wmt14 = load_dataset("wmt/wmt14", reversed_pair, split="train")
        dataset_lang_pair = reversed_pair
        print(f"Language pair {lang_pair} not found, using {reversed_pair} instead")
    except:
        print(f"Neither {lang_pair} nor {reversed_pair} found in WMT14 dataset")
        sys.exit(1)

def add_lengths(batch, src_lang, tgt_lang):
    src = [item[src_lang] for item in batch["translation"]]
    tgt = [item[tgt_lang] for item in batch["translation"]]
    src_ids = [tokenizer.encode(s).ids for s in src]
    tgt_ids = [tokenizer.encode(t).ids for t in tgt]
    return {
        "src_len": [len(x) for x in src_ids],
        "tgt_len": [len(x) for x in tgt_ids],
    }

# Save src and tgt token lengths
path_cache = f"./vocab/wmt14_{lang_pair}/train_len_cache"

ds = wmt14.map(add_lengths, batched=True, num_proc=4, fn_kwargs={"src_lang": src_lang, "tgt_lang": tgt_lang})
ds.select_columns(["src_len", "tgt_len"]).flatten_indices().save_to_disk(path_cache)

# Check lengths
ds_len = Dataset.load_from_disk(path_cache)
train_lengths = np.stack([ds_len["src_len"], ds_len["tgt_len"]], axis=1).astype("int32")
print(f"Loaded train_lengths from {path_cache} with shape {train_lengths.shape}")