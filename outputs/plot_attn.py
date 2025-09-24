import os
import sys
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tokenizers import Tokenizer

from src.model import Transformer
from src.utils import make_src_allow_mask, make_tgt_allow_mask

def plot_attention_heads(attn_maps, title, filename, src_tokens=None, tgt_tokens=None):
    num_heads = attn_maps.shape[0]
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_heads:
            im = ax.imshow(attn_maps[i], cmap='inferno', vmin=0.0, vmax=1.0)
            # ax.set_title(f"{title} - Head {i+1}", fontsize=10)

            # Set token labels on axes
            if src_tokens:
                ax.set_xticks(range(len(src_tokens)))
                ax.set_xticklabels(src_tokens, rotation=90, fontsize=10)
            if tgt_tokens:
                ax.set_yticks(range(len(tgt_tokens)))
                ax.set_yticklabels(tgt_tokens, fontsize=10)

            # # Add colorbar
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # cbar = plt.colorbar(im, cax=cax)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join("./outputs/attn_maps/per_layer", filename), dpi=300)
    plt.close()

def plot_single_attention(attn_map, title, filename, src_tokens=None, tgt_tokens=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn_map, cmap='inferno', vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12)

    # Set token labels on axes
    if src_tokens:
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90, fontsize=10)
    if tgt_tokens:
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=10)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join("./outputs/attn_maps/individual", filename), dpi=300)
    plt.close()

def visualize_attention(model, src, tgt, src_mask, tgt_mask, tokenizer):
    model.eval()
    
    # Setup directory
    os.makedirs("./outputs/attn_maps", exist_ok=True)
    os.makedirs("./outputs/attn_maps/individual", exist_ok=True)
    os.makedirs("./outputs/attn_maps/per_layer", exist_ok=True)
    
    with torch.no_grad():
        # Encode and decode to capture attention maps
        memory = model.encode(src, src_mask)
        model.decode(memory, src_mask, tgt, tgt_mask)

        # Decode token IDs to strings and split into tokens
        src_tokens = [tokenizer.id_to_token(id) for id in src[0].tolist()]
        tgt_tokens = [tokenizer.id_to_token(id) for id in tgt[0].tolist()]

        # Plot encoder self-attention
        for i, layer in enumerate(model.encoder.layers):
            attn_maps = layer.attention.attn[0].cpu().numpy()  # All heads
            plot_attention_heads(attn_maps, f"Encoder Layer {i+1} Self-Attention", f"s-attn_encoder_layer-{i+1:01d}.png", src_tokens, src_tokens)
            
            # Plot individual attention maps for each head
            for head in range(attn_maps.shape[0]):
                plot_single_attention(
                    attn_maps[head], 
                    f"Encoder L{i+1} H{head+1} Self-Attention", 
                    f"encoder_l{i+1:01d}_h{head+1:01d}_self.png", 
                    src_tokens, src_tokens
                )

        # Plot decoder self-attention
        for i, layer in enumerate(model.decoder.layers):
            attn_maps = layer.s_attention.attn[0].cpu().numpy()  # All heads
            plot_attention_heads(attn_maps, f"Decoder Layer {i+1} Self-Attention", f"s-attn_decoder_layer-{i+1:01d}.png", tgt_tokens, tgt_tokens)
            
            # Plot individual attention maps for each head
            for head in range(attn_maps.shape[0]):
                plot_single_attention(
                    attn_maps[head], 
                    f"Decoder L{i+1} H{head+1} Self-Attention", 
                    f"decoder_l{i+1:01d}_h{head+1:01d}_self.png", 
                    tgt_tokens, tgt_tokens
                )

        # Plot decoder cross-attention
        for i, layer in enumerate(model.decoder.layers):
            attn_maps = layer.x_attention.attn[0].cpu().numpy()  # All heads
            plot_attention_heads(attn_maps, f"Decoder Layer {i+1} Cross-Attention", f"x-attn_decoder_layer-{i+1:01d}.png", src_tokens, tgt_tokens)
            
            # Plot individual attention maps for each head
            for head in range(attn_maps.shape[0]):
                plot_single_attention(
                    attn_maps[head], 
                    f"Decoder L{i+1} H{head+1} Cross-Attention", 
                    f"decoder_l{i+1:01d}_h{head+1:01d}_cross.png", 
                    src_tokens, tgt_tokens
                )

# Config argparser
parser = argparse.ArgumentParser(description="Train the original Transformer for language translation")
parser.add_argument("--model_config", type=str, default="base", help="Size of model from configs.yaml ('base' or 'big')")
parser.add_argument("--lang", type=str, default="de", help="Langauge to translate to/from English ('cs', 'de', 'fr', 'hi', 'ru)")
args = parser.parse_args()

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    "de": "Der schnelle braune Fuchs springt über den faulen Hund",
    "en": "The quick brown fox jumps over the lazy dog"
}

src_text = example["de"]
tgt_text = example["en"]

src_enc = tokenizer.encode(src_text.lower(), add_special_tokens=False)
tgt_enc = tokenizer.encode(tgt_text.lower(), add_special_tokens=False)

src_ids = torch.tensor(src_enc.ids, dtype=torch.long, device=device).unsqueeze(0)
tgt_ids = torch.tensor(tgt_enc.ids, dtype=torch.long, device=device).unsqueeze(0)

src_mask = make_src_allow_mask(src_ids, pad_token_id)
tgt_mask = make_tgt_allow_mask(tgt_ids, pad_token_id)

print("Plotting attention maps...")
visualize_attention(model, src_ids, tgt_ids, src_mask, tgt_mask, tokenizer)
print("Done")