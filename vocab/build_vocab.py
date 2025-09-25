import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Parse arguments
parser = argparse.ArgumentParser(description="Build BPE tokenizer on whitespace-separated text")
parser.add_argument("--src_lang", type=str, default="de", help="Source language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--vocab_size", type=int, default=37000, help="Size of token vocabulary")
parser.add_argument("--min_frequency", type=int, default=2, help="Minimum frequency for merges")
args = parser.parse_args()

# Load dataset
src_lang = args.src_lang
tgt_lang = args.tgt_lang
lang_pair = f"{src_lang}-{tgt_lang}"

print(f"Loading {lang_pair} corpus...")
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
        exit(1)

root = f"./vocab/wmt14_{lang_pair}"
os.makedirs(root, exist_ok=True)

# Save en-de examples to files
path_a = os.path.join(root, f"train_{src_lang}.txt")
path_b = os.path.join(root, f"train_{tgt_lang}.txt")

if not os.path.exists(path_a) and not os.path.exists(path_b):
    with open(path_a, "w", encoding="utf-8") as f_a, open(path_b, "w", encoding="utf-8") as f_b:
        for example in tqdm(wmt14, total=len(wmt14), desc=f"Saving {lang_pair} examples to {root}"):
            f_a.write(example["translation"][src_lang].strip() + "\n")
            f_b.write(example["translation"][tgt_lang].strip() + "\n")

# Setup tokenizer and trainer
path_tok = os.path.join(root, f"bpe_{lang_pair}.json")

if os.path.exists(path_tok):
    tokenizer = Tokenizer.from_file(path_tok)
    print(f"Loaded tokenizer from {path_tok}")
else:
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train tokenizer
    print(f"Training tokenizer...")
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens
    )
    tokenizer.train(
        files=[path_a, path_b],
        trainer=trainer
    )

    # Setup post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))]
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    # Save tokenizer
    tokenizer.save(path_tok)
    print(f"Saved tokenizer to {path_tok}")

# Test tokenizer
text = {
    'en': "Hello. The book is on the table.",
    'cs': "Ahoj. Kniha je na stole.",
    'de': "Hallo. Das Buch liegt auf dem Tisch.",
    'fr': "Bonjour. Le livre est sur la table.",
    'hi': "नमस्ते। किताब मेज़ पर है।",
    'ru': "Привет. Книга на столе."
}

for lang in [src_lang, tgt_lang]:
    print(f"===== {lang} =====")

    enc = tokenizer.encode(text[lang])
    print("Tokens:", enc.tokens)
    print("IDs:", enc.ids)

    dec = tokenizer.decode(enc.ids)
    print("Decoded:", dec)