import os
import sys
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tokenizers import Tokenizer
from src.model import Transformer
from src.utils import make_src_allow_mask, make_tgt_allow_mask

def plot_encoder_attention_subplots(model, src, src_mask, tokenizer, layer_head_combinations, output_dir="./outputs/attn_maps"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Encode to capture attention maps
        memory = model.encode(src, src_mask)

        # Decode token IDs to strings and split into tokens
        src_tokens = [tokenizer.id_to_token(id) for id in src[0].tolist()]

        # Create subplots with shared y-axis and adjusted figure size
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        for ax, (layer_idx, head_idx) in zip(axes, layer_head_combinations):
            attn_map = model.encoder.layers[layer_idx].attention.attn[0, head_idx].cpu().numpy()
            im = ax.imshow(attn_map, cmap='inferno', vmin=0.0, vmax=1.0)
            ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}", fontsize=14)

            # Set token labels on axes
            ax.set_xticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=90, fontsize=12)
            ax.set_yticks(range(len(src_tokens)))
            ax.set_yticklabels(src_tokens, fontsize=12)

            # # Add colorbar
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "encoder_attention_subplots.png"), dpi=300)
        plt.close()

def plot_decoder_attention_subplots(model, src, tgt, src_mask, tgt_mask, tokenizer, self_layer_head_combinations, cross_layer_head_combinations, output_dir="./outputs/attn_maps"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Encode and decode to capture attention maps
        memory = model.encode(src, src_mask)
        model.decode(memory, src_mask, tgt, tgt_mask)

        # Decode token IDs to strings and split into tokens
        src_tokens = [tokenizer.id_to_token(id) for id in src[0].tolist()]
        tgt_tokens = [tokenizer.id_to_token(id) for id in tgt[0].tolist()]

        # Create subplots for decoder self-attention
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        for ax, (layer_idx, head_idx) in zip(axes, self_layer_head_combinations):
            attn_map = model.decoder.layers[layer_idx].s_attention.attn[0, head_idx].cpu().numpy()
            im = ax.imshow(attn_map, cmap='inferno', vmin=0.0, vmax=1.0)
            ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}", fontsize=14)

            # Set token labels on axes
            ax.set_xticks(range(len(tgt_tokens)))
            ax.set_xticklabels(tgt_tokens, rotation=90, fontsize=12)
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_yticklabels(tgt_tokens, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "decoder_self_attention_subplots.png"), dpi=300)
        plt.close()

        # Create subplots for decoder cross-attention
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)  # Adjusted figure size
        for ax, (layer_idx, head_idx) in zip(axes, cross_layer_head_combinations):
            attn_map = model.decoder.layers[layer_idx].x_attention.attn[0, head_idx].cpu().numpy()
            im = ax.imshow(attn_map, cmap='inferno', vmin=0.0, vmax=1.0)
            ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}", fontsize=14)

            # Set token labels on axes
            ax.set_xticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=90, fontsize=12)
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_yticklabels(tgt_tokens, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "decoder_cross_attention_subplots.png"), dpi=300)
        plt.close()

# Config argparser
parser = argparse.ArgumentParser(description="Train the original Transformer for language translation")
parser.add_argument("--model_config", type=str, default="base", help="Size of model from configs.yaml ('base' or 'big')")
parser.add_argument("--src_lang", type=str, default="en", help="Source language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--tgt_lang", type=str, default="de", help="Target language ('cs', 'de', 'fr', 'hi', 'ru)")
args = parser.parse_args()

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Create source and target tokens
max_len = 16
pad_token_id = tokenizer.token_to_id("<pad>")

example = {
    "cs": "Rychlá hnědá liška přeskočí líného psa",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund",
    "fr": "Le rapide renard brun saute par-dessus le chien paresseux",
    "hi": "तेज भूरी लोमड़ी सुस्त कुत्ते के ऊपर कूदती है",
    "ru": "Быстрая коричневая лиса перепрыгивает через ленивую собаку",
    "en": "The quick brown fox jumps over the lazy dog"
}

src_text = example[src_lang]
tgt_text = example[tgt_lang]

src_enc = tokenizer.encode(src_text.lower(), add_special_tokens=False)
tgt_enc = tokenizer.encode(tgt_text.lower(), add_special_tokens=False)

src_ids = torch.tensor(src_enc.ids, dtype=torch.long, device=device).unsqueeze(0)
tgt_ids = torch.tensor(tgt_enc.ids, dtype=torch.long, device=device).unsqueeze(0)

src_mask = make_src_allow_mask(src_ids, pad_token_id)
tgt_mask = make_tgt_allow_mask(tgt_ids, pad_token_id)

output_dir = f"./outputs/attn_maps/{lang_pair}/select"

# Example layer and head combinations for subplots
encoder_layer_head_combinations = [(0, 0), (2, 6), (5, 6)]  # Modify as needed
decoder_self_layer_head_combinations = [(0, 6), (2, 6), (4, 3)]  # Modify as needed
decoder_cross_layer_head_combinations = [(0, 6), (2, 6), (4, 6)]  # Modify as needed

print("Plotting encoder attention maps...")
plot_encoder_attention_subplots(model, src_ids, src_mask, tokenizer, encoder_layer_head_combinations, output_dir)
print("Plotting decoder self- and cross-attention maps...")
plot_decoder_attention_subplots(model, src_ids, tgt_ids, src_mask, tgt_mask, tokenizer, decoder_self_layer_head_combinations, decoder_cross_layer_head_combinations, output_dir)
print("Done")