import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Config argparser
parser = argparse.ArgumentParser(description="Plot loss curves")
parser.add_argument("--lang", type=str, default="de", help="Langauge to translate to/from English ('cs', 'de', 'fr', 'hi', 'ru)")
args = parser.parse_args()

def thousands(x, pos):
    return f"{int(x/1000)}K" if x >= 1000 else "0"

lang_a = args.lang
lang_b = "en"
lang_pair = f"{lang_a}-{lang_b}"
out_path = f"./outputs/loss_{lang_pair}.png"

files = {
    'train': f"./outputs/base-{lang_pair}-train.csv",
    'valid': f"./outputs/base-{lang_pair}-valid.csv"
}

for k, v in files.items():
    df = pd.read_csv(v)
    plt.plot(df['Step'], df['Value'], label=k)

plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig(out_path, dpi=300)

print(f"Saved plot to {out_path}")