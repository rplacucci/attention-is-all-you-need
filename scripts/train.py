import os
import sys
import time
import yaml
import argparse

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer

from src.model import Transformer
from src.utils import LabelSmoothing, greedy_decode
from src.dataset import WMT14Dataset

# torchrun --standalone -nproc-per-node 4 train.py

# Config argparser
parser = argparse.ArgumentParser(description="Train the original Transformer for language translation")
parser.add_argument("--model_config", type=str, default="base", help="Size of model from configs.yaml ('base' or 'big')")
parser.add_argument("--lang", type=str, default="de", help="Langauge to translate to/from English ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per training step (16, 32, 64, etc.)")
args = parser.parse_args()

# Initialize distributed processing
distributed = int(os.environ.get("RANK", -1)) != -1

if distributed:
    assert torch.cuda.is_available(), "CUDA required for distributed processing."
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_process = rank == 0
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=local_rank,
    )
else:
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision("high")

# Setup tokenizer
lang_a = args.lang
lang_b = "en"
lang_pair = f"{lang_a}-{lang_b}"

path_vocab = f"./vocab/wmt14_{lang_pair}/bpe_{lang_pair}.json"
if not os.path.exists(path_vocab):
    print(f"Tokenizer file {path_vocab} does not exist. Run 'python -m vocab.build_vocab' and try again")
    sys.exit(1)
tokenizer = Tokenizer.from_file(path_vocab)
print(f"Loaded tokenizer from {path_vocab}")