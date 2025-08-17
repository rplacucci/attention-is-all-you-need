from abc import ABC, abstractmethod

class Scheduler(ABC):
    """Abstract base class for custom learning rate schedulers."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    @abstractmethod
    def step(self):
        """Advance the scheduler by one step and update the optimizer LRs."""
        pass

    def get_lr(self):
        """Returns the current learning rates."""
        return self._last_lr

    def state_dict(self):
        """Returns the scheduler state for checkpointing."""
        return {"last_lr": self._last_lr}

    def load_state_dict(self, state_dict):
        """Restores scheduler state from a checkpoint."""
        self._last_lr = state_dict["last_lr"]
        for lr, group in zip(self._last_lr, self.optimizer.param_groups):
            group["lr"] = lr

class InverseSqrtLR(Scheduler):
    """
    Learning rate scheduler from Vaswani et al. (2017):
    lr = embed_size^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    def __init__(self, optimizer, embed_size, warmup_steps):
        super().__init__(optimizer)
        self.embed_size = embed_size
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.scale_factor = embed_size ** -0.5
        self.base_lrs = [1.0 for _ in optimizer.param_groups]  # Start from 1.0 and scale

    def step(self):
        self.step_num += 1
        step = self.step_num
        warmup = self.warmup_steps

        lr = self.scale_factor * min(step ** -0.5, step * (warmup ** -1.5))
        lrs = [base_lr * lr for base_lr in self.base_lrs]

        for group, lr_val in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr_val
        self._last_lr = lrs

    def state_dict(self):
        base_state = super().state_dict()
        base_state.update({
            "step_num": self.step_num,
            "embed_size": self.embed_size,
            "warmup_steps": self.warmup_steps,
            "base_lrs": self.base_lrs
        })
        return base_state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.step_num = state_dict["step_num"]
        self.embed_size = state_dict["embed_size"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.base_lrs = state_dict["base_lrs"]