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
lang_pairs = [f"{lang_a}-{lang_b}", f"{lang_b}-{lang_a}"]
out_path = f"./outputs/loss_{lang_a}.png"

for lang_pair in lang_pairs:
    input_path = f"./outputs/base-{lang_pair}-valid.csv"
    df = pd.read_csv(input_path)
    plt.plot(df['Step'], df['Value'], label=lang_pair)
plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
plt.xlabel("Step")
plt.ylabel("Loss")
plt.ylim([2.5, 4.5])
plt.legend()
plt.grid(True)
plt.savefig(out_path, dpi=300)

print(f"Saved plot to {out_path}")