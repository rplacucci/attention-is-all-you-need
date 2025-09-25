import os
import sys
import json
import yaml
import random
import argparse

import torch
import evaluate
from torch.utils.data import DataLoader

from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from sacremoses import MosesDetokenizer

from src.model import Transformer
from src.collate import pad_collate
from src.dataset import WMT14Dataset
from src.utils import make_src_allow_mask
from src.decode import beam_search_decode

# torchrun --standalone --nproc-per-node=1 -m scripts.test

# Config argparser
parser = argparse.ArgumentParser(description="Test the original Transformer for language translation")
parser.add_argument("--model_config", type=str, default="base", help="Size of model from configs.yaml ('base' or 'big')")
parser.add_argument("--src_lang", type=str, default="de", help="Source language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per testing step (16, 32, 64, etc.)")
parser.add_argument("--max_len", type=int, default=128, help="Maximum number of tokens to decode (default: 100, as in Vaswani et al.)")
args = parser.parse_args()

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision("high")

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

# Get special token ids
pad_token_id = tokenizer.token_to_id("<pad>")
sos_token_id = tokenizer.token_to_id("<s>")
eos_token_id = tokenizer.token_to_id("</s>")

# Setup model
with open("./scripts/configs.yaml", "r") as f:
    config = yaml.safe_load(f)[args.model_config]

model = Transformer(vocab_size=vocab_size, **config).to(device)
model_dir = f"./models/{args.model_config}-{lang_pair}"
model_path = os.path.join(model_dir, "model.pt") 

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
print(f"Loaded {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters from {model_path} to {device}")

# Get maximim sequence length and batch size
batch_size = args.batch_size
max_len = args.max_len

# Prepare collate function
collate_fn = lambda batch: pad_collate(batch, pad_id=pad_token_id)

# Load test dataset
try:
    wmt14 = load_dataset("wmt/wmt14", lang_pair, split='test')
    dataset_lang_pair = lang_pair
except:
    # If the pair doesn't exist, try reversing it (e.g., "en-de" -> "de-en")
    reversed_pair = f"{tgt_lang}-{src_lang}"
    try:
        wmt14 = load_dataset("wmt/wmt14", reversed_pair, split='test')
        dataset_lang_pair = reversed_pair
        print(f"Language pair {lang_pair} not found, using {reversed_pair} instead")
    except:
        print(f"Neither {lang_pair} nor {reversed_pair} found in WMT14 dataset")
        sys.exit(1)

print(f"Loaded WMT14[{dataset_lang_pair}] dataset with {len(wmt14):,} test sequence pairs")

dataset = WMT14Dataset(
    dataset=wmt14,
    tokenizer=tokenizer,
    max_len=max_len,
    src_lang=src_lang,
    tgt_lang=tgt_lang
)

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    shuffle=False
)

# Setup detokenizer
detokenizer = MosesDetokenizer(lang=tgt_lang)

# Setup directories for saving
out_dir = f"./results"
os.makedirs(out_dir, exist_ok=True)

# Begin testing
preds = []
refs = []

model.eval()
with torch.no_grad():
    for idx, batch in enumerate(tqdm(loader, total=len(loader), desc="Generating predictions")):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        src = batch['src_ids']
        src_mask = make_src_allow_mask(src, pad_token_id)

        # Max output length: input length + 50 (batch-wise, using max src length)
        src_nonpad_lens = (src != pad_token_id).sum(dim=1)
        max_src_len = int(src_nonpad_lens.max().item())
        decode_max_len = max_src_len + 50

        # Beam search decoding
        decoded = beam_search_decode(model, src, src_mask, decode_max_len, device, sos_token_id, eos_token_id, pad_token_id)

        # Remove start token and everything after end token
        decoded_text = []
        for seq in decoded:
            tokens = seq.tolist()
            if eos_token_id in tokens:
                tokens = tokens[:tokens.index(eos_token_id)]
            decoded_text.append(tokenizer.decode(tokens, skip_special_tokens=True))
        preds.extend(decoded_text)

        # Reference sentences
        tgt_ids = batch['tgt_ids']
        decoded_text = tokenizer.decode_batch(tgt_ids.tolist())
        refs.extend([[ref] for ref in decoded_text])

# Detokenize predictions and references for sacreBLEU
detokenized_preds = [detokenizer.detokenize(pred.split()) for pred in preds]
detokenized_refs = [detokenizer.detokenize(ref[0].split()) for ref in refs]

# Setup scoring metric
bleu = evaluate.load('sacrebleu')

# Evaluate and print
score = bleu.compute(predictions=detokenized_preds, references=detokenized_refs)
print(score)

# Save
with open(os.path.join(out_dir, f"{args.model_config}-{lang_pair}.json"), "w") as f:
    json.dump(score, f, indent=4)