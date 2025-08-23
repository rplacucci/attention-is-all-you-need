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
from torch.optim import Adam

from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer

from src.model import Transformer
from src.utils import LabelSmoothing, greedy_decode
from src.dataset import WMT14Dataset
from src.scheduler import InverseSqrtLR

# torchrun --standalone --nproc-per-node=4 -m scripts.train

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
vocab_size = tokenizer.get_vocab_size()
if master_process:
    print(f"Loaded tokenizer from {path_vocab} with size {vocab_size:,}")

# Setup model
with open("./scripts/configs.yaml", "r") as f:
    config = yaml.safe_load(f)[args.model_config]

model = Transformer(vocab_size=vocab_size, **config).to(device)
if master_process:
    print(f"Loaded {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Wrap the model with DistributedDataParallel if distributed training is enabled
if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(f"Initialized model on {device}")
time.sleep(0.5)

# Get maximim sequence length and batch size
batch_size = args.batch_size
max_len = config['max_len']

# Load dev dataset
wmt14 = load_dataset("wmt/wmt14", lang_pair)
splits = ['train', 'validation']
time.sleep(5)

if master_process:
    print(f"Loaded WMT14[{lang_pair}] dataset with {sum([len(wmt14[split]) for split in splits]):,} train/val sequence pairs")
if distributed:
    dist.barrier()

train_dataset = WMT14Dataset(
    dataset=wmt14['train'],
    tokenizer=tokenizer,
    max_len=max_len,
    src_lang=lang_a,
    tgt_lang=lang_b
)

train_sampler = DistributedSampler(
    dataset=train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0,
    prefetch_factor=None
)

train_dataiter = iter(train_dataloader)

valid_dataset = WMT14Dataset(
    dataset=wmt14['validation'],
    tokenizer=tokenizer,
    max_len=max_len,
    src_lang=lang_a,
    tgt_lang=lang_b
)

valid_sampler = DistributedSampler(
    dataset=valid_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed
)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=0,
    prefetch_factor=None
)

valid_dataiter = iter(valid_dataloader)

# Define number of training iterations
total_steps = 300000 if args.model_config == 'big' else 100000
if master_process:
    print(f"Total training steps set to {total_steps:,}")

# Define optimizer and learning rate schedule
betas = (0.9, 0.98)
eps = 1e-9
warmup_steps = 4000

optimizer = Adam(model.parameters(), lr=1.0, betas=betas, eps=eps, fused=True)
scheduler = InverseSqrtLR(optimizer=optimizer, embed_size=config['embed_size'], warmup_steps=warmup_steps)

if master_process:
    print(f"Loaded {optimizer.__class__.__name__} optimizer")
    print(f"Loaded {scheduler.__class__.__name__} learning rate scheduler")

# Setup tensorboard and directories for logging/saving
out_dir = f"./models/transformer-{args.model_config}-{lang_pair}"
log_dir = f"./logs/transformer-{args.model_config}-{lang_pair}"

if master_process:
    writer = SummaryWriter(log_dir)
    os.makedirs(out_dir)

if distributed:
    dist.barrier()