import math
import torch
import random

from dataclasses import dataclass
from torch.utils.data import Sampler

@dataclass
class BucketConfig:
    """
    Configuration for batching samples into buckets.

    Attributes:
        max_tokens (int): Maximum number of tokens per batch.
        max_padded_tokens (int): Maximum number of padded tokens per batch.
        bucket_size (int): Number of samples to consider in each local bucket.
        drop_last (bool): Whether to drop the last batch if it's smaller than bucket_size.
        allow_overshoot (int): Allowed overshoot for token limits.
        seed (int): Random seed for reproducibility.
    """
    max_tokens: int = 25000
    max_padded_tokens: int = 24000
    bucket_size: int = 2048
    drop_last: bool = True
    allow_overshoot: int = 512
    seed: int = 42

class BucketPacker:
    """
    Packs indices into batches with approximately max_tokens.
    Uses local sorting within buckets to reduce padding.

    Args:
        lengths (list[tuple[int, int]]): List of (src_len, tgt_len) tuples.
        cfg (BucketConfig): Configuration for bucketing.
    """
    def __init__(self, lengths, cfg):
        # Store list of (src_len, tgt_len) tuples
        self.lengths = lengths

        # Unpack configuration
        self.max_tokens = cfg.max_tokens
        self.max_padded_tokens = cfg.max_padded_tokens
        self.bucket_size = cfg.bucket_size
        self.drop_last = cfg.drop_last
        self.allow_overshoot = cfg.allow_overshoot
        self.seed = cfg.seed

    def pack(self, indices):
        """
        Packs indices into batches, trying to keep each batch within token limits.

        Args:
            indices (list[int]): List of sample indices to batch.

        Yields:
            list[int]: Batch of indices.
        """
        for i in range(0, len(indices), self.bucket_size):
            # Take a local bucket of indices
            bucket = indices[i:i+self.bucket_size]
            # Sort bucket by max(src_len, tgt_len) to minimize padding
            bucket.sort(key=lambda j: max(self.lengths[j][0], self.lengths[j][1]))

            batch = []
            batch_src_tokens = 0
            batch_tgt_tokens = 0
            batch_sum_tokens = 0

            for j in bucket:
                sl, tl = self.lengths[j]
                # Calculate new max lengths if this sample is added
                new_batch_src_tokens = max(batch_src_tokens, sl)
                new_batch_tgt_tokens = max(batch_tgt_tokens, tl)

                # Calculate padded and sum token counts if added
                padded_if_added = (len(batch) + 1) * (new_batch_src_tokens + new_batch_tgt_tokens)
                sum_if_added = batch_sum_tokens + sl + tl

                # Check if adding this sample stays within limits
                if (
                    sum_if_added <= self.max_tokens + self.allow_overshoot
                    and
                    padded_if_added <= self.max_padded_tokens + self.allow_overshoot
                ):
                    # Safe to add sample to batch
                    batch.append(j)
                    batch_src_tokens = new_batch_src_tokens
                    batch_tgt_tokens = new_batch_tgt_tokens
                    batch_sum_tokens = sum_if_added
                else:
                    # Flush current batch and start a new one
                    if batch:
                        yield batch
                    batch = [j]
                    batch_src_tokens = sl
                    batch_tgt_tokens = tl
                    batch_sum_tokens = sl + tl

            # Yield last batch if not dropping incomplete batches
            if batch and not self.drop_last:
                yield batch

class BucketSampler(Sampler):
    """
    Sampler that yields batches of indices packed according to token limits.

    Args:
        lengths (list[tuple[int, int]]): List of (src_len, tgt_len) tuples.
        cfg (BucketConfig): Configuration for bucketing.
    """
    def __init__(self, lengths, cfg):
        # Store list of (src_len, tgt_len) tuples
        self.lengths = lengths
        self.epoch_indices = None
        self.packer = BucketPacker(lengths, cfg)

        # Unpack configuration
        self.max_tokens = cfg.max_tokens
        self.max_padded_tokens = cfg.max_padded_tokens
        self.bucket_size = cfg.bucket_size
        self.drop_last = cfg.drop_last
        self.allow_overshoot = cfg.allow_overshoot
        self.seed = cfg.seed

    def set_epoch_indices(self, epoch_indices, shuffle=True):
        """
        Set the indices for the current epoch and optionally shuffle them.

        Args:
            epoch_indices (list[int]): Indices to use for this epoch.
            shuffle (bool): Whether to shuffle the indices.
        """
        self.epoch_indices = list(epoch_indices)
        if shuffle:
            generator = random.Random(self.seed)
            generator.shuffle(self.epoch_indices)

    def __iter__(self):
        """
        Yield batches of indices for the current epoch.

        Yields:
            list[int]: Batch of indices.
        """
        if self.epoch_indices is None:
            raise RuntimeError("BucketSampler: call set_epoch_indices() first")
        yield from self.packer.pack(self.epoch_indices)

    def __len__(self):
        """
        Return the number of batches for the current epoch.

        Returns:
            int: Number of batches.
        """
        if self.epoch_indices is None:
            return 0
        num_batches = 0
        batch_src_tokens = 0
        batch_tgt_tokens = 0
        batch_sum_tokens = 0
        batch_size = 0

        # Iterate through indices and count batches using the same logic as pack()
        for j in self.epoch_indices:
            sl, tl = self.lengths[j]
            new_batch_src_tokens = max(batch_src_tokens, sl)
            new_batch_tgt_tokens = max(batch_tgt_tokens, tl)

            padded_if_added = (batch_size + 1) * (new_batch_src_tokens + new_batch_tgt_tokens)
            sum_if_added = batch_sum_tokens + sl + tl

            if (
                sum_if_added <= self.max_tokens + self.allow_overshoot
                and
                padded_if_added <= self.max_padded_tokens + self.allow_overshoot
            ):
                # Safe to add sample to batch
                batch_size += 1
                batch_src_tokens = new_batch_src_tokens
                batch_tgt_tokens = new_batch_tgt_tokens
                batch_sum_tokens = sum_if_added
            else:
                # Start a new batch
                num_batches += 1
                batch_size = 1
                batch_src_tokens = sl
                batch_tgt_tokens = tl
                batch_sum_tokens = sl + tl

        # Account for the last batch if it exists
        if batch_size:
            num_batches += 1
        
        return num_batches