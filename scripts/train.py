import os
import sys
import time
import yaml
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from tqdm import tqdm
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer

from src.model import Transformer
from src.utils import make_src_allow_mask, make_tgt_allow_mask
from src.dataset import WMT14Dataset
from src.scheduler import InverseSqrtLR
from src.bucket import BucketConfig, BucketSampler
from src.collate import pad_collate

# torchrun --standalone --nproc-per-node=4 -m scripts.train

# Config argparser
parser = argparse.ArgumentParser(description="Train the original Transformer for language translation")
parser.add_argument("--model_config", type=str, default="base", help="Size of model from configs.yaml ('base' or 'big')")
parser.add_argument("--src_lang", type=str, default="de", help="Source language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language ('cs', 'de', 'fr', 'hi', 'ru)")
parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per training step (16, 32, 64, etc.)")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation steps to use a certain number of non-pad tokens per iteration")
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
src_lang = args.src_lang
tgt_lang = args.tgt_lang
lang_pair = f"{src_lang}-{tgt_lang}"

path_vocab = f"./vocab/wmt14_{lang_pair}/bpe_{lang_pair}.json"
if not os.path.exists(path_vocab) and master_process:
    print(f"Tokenizer file {path_vocab} does not exist. Run 'python -m vocab.build_vocab' and try again")
    destroy_process_group()
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

# Get maximum sequence length and batch size
batch_size = args.batch_size
max_len = config['max_len']

# Load dev dataset
try:
    wmt14 = load_dataset("wmt/wmt14", lang_pair)
    dataset_lang_pair = lang_pair
except:
    # If the pair doesn't exist, try reversing it (e.g., "en-de" -> "de-en")
    reversed_pair = f"{tgt_lang}-{src_lang}"
    try:
        wmt14 = load_dataset("wmt/wmt14", reversed_pair)
        dataset_lang_pair = reversed_pair
        if master_process:
            print(f"Language pair {lang_pair} not found, using {reversed_pair} instead")
    except:
        if master_process:
            print(f"Neither {lang_pair} nor {reversed_pair} found in WMT14 dataset")
        if distributed:
            destroy_process_group()
        sys.exit(1)

splits = ['train', 'validation']
time.sleep(5)

if master_process:
    print(f"Loaded WMT14[{lang_pair}] dataset with {sum([len(wmt14[split]) for split in splits]):,} train/val sequence pairs")
if distributed:
    dist.barrier()

pad_token_id = tokenizer.token_to_id("<pad>")
collate_fn = lambda batch: pad_collate(batch, pad_id=pad_token_id)

train_dataset = WMT14Dataset(
    dataset=wmt14['train'],
    tokenizer=tokenizer,
    max_len=max_len,
    src_lang=src_lang,
    tgt_lang=tgt_lang
)

train_sampler = StatefulDistributedSampler(
    dataset=train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed
)

path_cache = f"./vocab/wmt14_{lang_pair}/train_len_cache"
if not os.path.exists(path_cache) and master_process:
    print(f"Token lengths cache file {path_cache} does not exist. Run 'python -m vocab.survey' and try again")
    destroy_process_group()
    sys.exit(1)

ds_len = Dataset.load_from_disk(path_cache)
train_lengths = np.stack([ds_len["src_len"], ds_len["tgt_len"]], axis=1).astype("int32")
time.sleep(3)

if master_process:
    print(f"Loaded train_lengths from {path_cache} with {train_lengths.shape[0]:,} train sequence pairs")
if distributed:
    dist.barrier()

bucket_cfg = BucketConfig(
    max_tokens=25000,
    max_padded_tokens=24000,
    bucket_size=4096,
    drop_last=True,
    allow_overshoot=512,
    seed=42
)

bucket_sampler = BucketSampler(lengths=train_lengths, cfg=bucket_cfg)

train_dataloader = StatefulDataLoader(
    dataset=train_dataset,
    batch_sampler=bucket_sampler,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    collate_fn=collate_fn
)

valid_dataset = WMT14Dataset(
    dataset=wmt14['validation'],
    tokenizer=tokenizer,
    max_len=max_len,
    src_lang=src_lang,
    tgt_lang=tgt_lang
)

valid_sampler = StatefulDistributedSampler(
    dataset=valid_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    seed=seed
)

valid_dataloader = StatefulDataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=0,
    prefetch_factor=None,
    collate_fn=collate_fn
)

# Define number of training iterations
total_steps = 300000 if args.model_config == 'big' else 100000
if master_process:
    print(f"Total training steps set to {total_steps:,}")

# Set gradient accumulation steps
grad_accum_steps = args.grad_accum_steps
if master_process:
    print(f"Grad accumulation steps set to {grad_accum_steps}")

# Define loss criterion
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)

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
out_dir = f"./models/{args.model_config}-{lang_pair}"
log_dir = f"./logs/{args.model_config}-{lang_pair}"

if master_process:
    writer = SummaryWriter(log_dir)
    os.makedirs(out_dir, exist_ok=True)

if distributed:
    dist.barrier()

# Resume from checkpoint if available
epoch = 0
start_step = 0
ckpt_main_path = os.path.join(out_dir, "ckpt.pt")
ckpt_rank_path = os.path.join(out_dir, f"ckpt_rank{rank}.pt")

if os.path.exists(ckpt_main_path):
    if master_process:
        print(f"Found model checkpoint at {ckpt_main_path}")

    state = torch.load(ckpt_main_path, map_location=device)

    model.module.load_state_dict(state['model']) if distributed else model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    # restore bookkeeping
    epoch = state['epoch']
    start_step = state['step'] + 1
    loss = state['loss']

    if master_process:
        print(f"Resuming from step {start_step}")

    # broadcast loaded weights to all ranks
    if distributed:  
        for param in model.parameters():  
            dist.broadcast(param.data, src=0)
        dist.barrier()

if os.path.exists(ckpt_rank_path):
    print(f"Found loader checkpoint at {ckpt_rank_path}")
    loader_state = torch.load(ckpt_rank_path, map_location=device)

    epoch = loader_state['epoch']
    start_step = loader_state['step'] + 1

    # Rebuild train sampler before loading state
    train_sampler.set_epoch(epoch)
    bucket_sampler.set_epoch_indices(list(iter(train_sampler)), shuffle=False)

    train_dataloader.load_state_dict(loader_state['train'])
    valid_dataloader.load_state_dict(loader_state['valid'])
else:
    epoch = 0
    train_sampler.set_epoch(epoch)
    bucket_sampler.set_epoch_indices(list(iter(train_sampler)), shuffle=False)
    
time.sleep(3)
if distributed:
    dist.barrier()

# Begin training loop
log_steps = 10
ckpt_steps = 100
save_steps = 10000 if args.model_config == 'base' else 100000
valid_steps = 200
step = start_step

train_dataiter = iter(train_dataloader)
while step < total_steps:
    # Validate model
    if step % valid_steps == 0:
        model.eval()
        val_loss_accum = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for batch in valid_dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                src, tgt = batch["src_ids"], batch["tgt_ids"]

                src_mask = make_src_allow_mask(src, pad_token_id)
                tgt_x = tgt[:, :-1]
                tgt_y = tgt[:,  1:]
                tgt_x_mask = make_tgt_allow_mask(tgt_x, pad_token_id)

                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
                    logits = model(src, tgt_x, src_mask, tgt_x_mask)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_y.reshape(-1)
                )

                loss /= len(valid_dataloader)
                val_loss_accum += loss.detach()

        if distributed:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        val_loss = val_loss_accum.item()

        if master_process:
            print(f"(valid) step: {step:6d}/{total_steps} | loss: {val_loss:.4f}")
            writer.add_scalar("loss/valid", val_loss, step)

    # Train model
    model.train()
    start = time.time()
    optimizer.zero_grad()

    loss_accum = torch.tensor(0.0, device=device)
    non_pad_tokens = 0
    for accum_step in range(grad_accum_steps):
        try:
            batch = next(train_dataiter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            bucket_sampler.set_epoch_indices(list(iter(train_sampler)), shuffle=False)
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        src, tgt = batch["src_ids"], batch["tgt_ids"]

        src_mask = make_src_allow_mask(src, pad_token_id)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:,  1:]
        tgt_x_mask = make_tgt_allow_mask(tgt_x, pad_token_id)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
            logits = model(src, tgt_x, src_mask, tgt_x_mask)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_y.reshape(-1)
        )

        loss /= grad_accum_steps
        loss_accum += loss.detach()
        if distributed:
            model.require_backward_grad_sync = (accum_step == grad_accum_steps - 1)
        loss.backward()

        src_tokens = int(batch['src_mask'].sum().item())
        tgt_tokens = int(batch['tgt_mask'].sum().item())
        non_pad_tokens += (src_tokens + tgt_tokens)

    if distributed:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scheduler.step()
    optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tokens_per_sec = int(batch_size * max_len * grad_accum_steps * world_size / elapsed) if elapsed > 0 else 0

    loss = loss_accum.item()
    lr = scheduler.get_lr()[0]

    if master_process:
        print(f"(train) step: {step:6d}/{total_steps} | loss: {loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | tok: {non_pad_tokens:,} | tok/sec: {tokens_per_sec:07,d}")
        if step % log_steps == 0:
            writer.add_scalar("loss/train", loss, step)
            writer.add_scalar("lr", lr, step)

    if distributed:
        dist.barrier()

    # Save checkpoint
    if step % ckpt_steps == 0 and step > 0:
        state = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'model': model.module.state_dict() if distributed else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if master_process:
            torch.save(state, ckpt_main_path)
            print(f"Saved model checkpoint to {ckpt_main_path}")

        time.sleep(1)
        if distributed:
            dist.barrier()
        
        loader_state = {
            'epoch': epoch,
            'step': step,
            'train': train_dataloader.state_dict(),
            'valid': valid_dataloader.state_dict()
        }
        torch.save(loader_state, ckpt_rank_path)
        print(f"Saved loader checkpoint to {ckpt_rank_path} on {device}")

        time.sleep(1)
        if distributed:
            dist.barrier()

    # Save intermediate model
    if step % save_steps == 0 and step > 0:
        if master_process:
            torch.save(
                model.module.state_dict() if distributed else model.state_dict(),
                os.path.join(out_dir, f"model_{step:6d}.pt")
            )
            print(f"Saved model to {out_dir}")

    step += 1

# Save final model
if master_process:
    torch.save(
        model.module.state_dict() if distributed else model.state_dict(),
        os.path.join(out_dir, "model.pt")
    )
    print(f"Saved final model to {out_dir}")

if distributed:
    dist.barrier()

# Cleanup distributed processing
if master_process:
    print("Cleaning up...")

    if writer is not None:
        writer.flush()
        writer.close()

del model, optimizer, scheduler
del train_dataset, train_sampler, train_dataloader
del valid_dataset, valid_sampler, valid_dataloader
del tokenizer, writer

torch.cuda.empty_cache()

if distributed:
    torch.cuda.set_device(local_rank)
    dist.barrier()
    destroy_process_group()

if master_process:
    print("Goodbye!")

time.sleep(1)