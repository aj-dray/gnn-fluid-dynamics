"""
Simplified learning rate scheduler implementation for FVGN training.
"""
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, CosineAnnealingLR
import math

class StepThenDecay(_LRScheduler):
    """
    Two-stage learning rate scheduler:
    1. Step decay at milestone (reduces LR by gamma factor)
    2. Exponential decay after a specified number of mini-epochs

    Args:
        optimizer: PyTorch optimizer
        config: Training config object
        total_mini_epochs: Total number of mini-epochs for training
    """

    def __init__(self, optimizer, config, total_mini_epochs, last_epoch=-1):
        self.milestone = int((config.lr_ms1) * total_mini_epochs)
        self.milestone_gamma = config.lr_ms1_gamma
        self.exp_decay_start = int((config.lr_ms2) * total_mini_epochs) if config.lr_ms2 else total_mini_epochs
        self.exp_gamma = config.lr_ms2_gamma
        self.decay_steps = total_mini_epochs - self.exp_decay_start

        min_lr = config.lr_min or 1e-6

        # Handle min_lr as single value or list
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        # For tracking milestone hit
        self.milestone_applied = False

        # Determine warm-up period (ratio of total mini-epochs)
        self.warmup_steps = int((config.lr_wu or 0) * total_mini_epochs)

        super(StepThenDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on current epoch."""
        t = self.last_epoch  # current “epoch”/step

                # —— Warm-up stage ——
        if self.warmup_steps > 0 and t < self.warmup_steps:
            # scale from 0 → base_lr
            return [
                base_lr * (t / float(max(1, self.warmup_steps)))
                for base_lr in self.base_lrs
            ]

        if t <= self.milestone:
            # stage‑1 : keep the base learning rate
            return [base_lr for base_lr in self.base_lrs]

        elif t <= self.exp_decay_start:
            # stage‑2 : single step decay (base_lr * milestone_gamma)
            return [base_lr * self.milestone_gamma for base_lr in self.base_lrs]

        # stage‑3 : exponential decay every `decay_steps`
        elapsed = t - self.exp_decay_start
        factor = self.exp_gamma ** (elapsed / self.decay_steps)

        return [
            min_lr
            + max(base_lr * self.milestone_gamma - min_lr, 0.0) * factor
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

    def state_dict(self):
        """Return state dict for checkpointing."""
        state = super(StepThenDecay, self).state_dict()
        state.update({
            'milestone': self.milestone,
            'milestone_gamma': self.milestone_gamma,
            'exp_decay_start': self.exp_decay_start,
            'exp_gamma': self.exp_gamma,
            'min_lrs': self.min_lrs,
            'milestone_applied': self.milestone_applied
        })
        return state

    def load_state_dict(self, state_dict):
        """Load state dict for resuming training."""
        self.milestone = state_dict.pop('milestone')
        self.milestone_gamma = state_dict.pop('milestone_gamma')
        self.exp_decay_start = state_dict.pop('exp_decay_start')
        self.exp_gamma = state_dict.pop('exp_gamma')
        self.min_lrs = state_dict.pop('min_lrs')
        self.milestone_applied = state_dict.pop('milestone_applied')
        super(StepThenDecay, self).load_state_dict(state_dict)

class OneCycle(_LRScheduler):
    """
    Wrapper for PyTorch's OneCycleLR to match custom scheduler interface.
    """
    def __init__(self, optimizer, config, total_mini_epochs, last_epoch=-1):
        max_lr = config.lr_max
        pct_start = config.lr_wu or 0.2
        anneal_strategy = "cos"
        div_factor = 1 / (config.lr_wu_gamma or 0.04)
        final_div_factor = 1 / (config.lr_ms1_gamma or 1e-4)
        three_phase = False

        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_mini_epochs,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch
        )
        # For compatibility with _LRScheduler
        self.optimizer = optimizer
        self.base_lrs = self.scheduler.base_lrs
        self.last_epoch = self.scheduler.last_epoch

    def get_lr(self):
        # Delegate to the wrapped scheduler
        return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        self.scheduler.step(epoch)
        self.last_epoch = self.scheduler.last_epoch

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class CosineAnnealing(_LRScheduler):
    """
    Cosine Annealing LR with optional linear warmup phase.
    """
    def __init__(self, optimizer, config, total_mini_epochs, last_epoch=-1):
        warmup_steps = int((config.lr_wu or 0) * total_mini_epochs)
        max_lr = config.lr_max
        min_lr = config.lr_min or 0

        self.warmup_steps = warmup_steps
        self.total_steps = total_mini_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

        # Handle max_lr as single value or list
        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        # Handle min_lr as single value or list
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        # Set base_lrs to max_lrs for compatibility with _LRScheduler
        self.base_lrs = self.max_lrs.copy()

        # Create cosine scheduler that will be used after warmup
        # We'll manually set the optimizer's base_lrs to max_lrs for the cosine scheduler
        temp_base_lrs = [group['lr'] for group in optimizer.param_groups]
        for group, max_lr_val in zip(optimizer.param_groups, self.max_lrs):
            group['lr'] = max_lr_val

        self.after_warmup_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_mini_epochs - warmup_steps,
            eta_min=min_lr
        )

        # Restore original learning rates
        for group, orig_lr in zip(optimizer.param_groups, temp_base_lrs):
            group['lr'] = orig_lr

        # For compatibility with _LRScheduler
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup from 0 to max_lr
            if self.warmup_steps == 0:
                return self.max_lrs
            warmup_factor = float(self.last_epoch + 1) / float(self.warmup_steps)
            return [max_lr * warmup_factor for max_lr in self.max_lrs]
        else:
            # Cosine annealing after warmup
            return self.after_warmup_scheduler.get_last_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch < self.warmup_steps:
            # During warmup
            self.last_epoch = epoch
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            # After warmup - use cosine annealing
            cosine_epoch = epoch - self.warmup_steps
            self.after_warmup_scheduler.last_epoch = cosine_epoch - 1  # -1 because step() will increment
            self.after_warmup_scheduler.step()
            self.last_epoch = epoch

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            'after_warmup_scheduler': self.after_warmup_scheduler.state_dict(),
            'max_lrs': self.max_lrs,
            'min_lrs': self.min_lrs,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.after_warmup_scheduler.load_state_dict(state_dict['after_warmup_scheduler'])
        self.max_lrs = state_dict.get('max_lrs', self.max_lrs)
        self.min_lrs = state_dict.get('min_lrs', self.min_lrs)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.total_steps = state_dict.get('total_steps', self.total_steps)


class ExponentialDecay(_LRScheduler):
    """
    Simple exponential decay learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        config: Training config object (expects lr_m1_decay)
        total_mini_epochs: Total number of mini-epochs for training
    """
    def __init__(self, optimizer, config, total_mini_epochs, last_epoch=-1):
        self.decay = config.lr_ms1_gamma
        self.total_mini_epochs = total_mini_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Exponential decay: lr = base_lr * decay^epoch
        return [
            base_lr * (self.decay ** self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def state_dict(self):
        return super().state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

class CosineAnnealingTwoPhase(_LRScheduler):
    """
    Five-phase LR schedule (four if lr_ms3 not set):
    1. Warmup: from lr_wu_gamma * lr_max to lr_max over lr_wu fraction.
    2. Hold: hold at lr_max until lr_ms1 fraction.
    3. Cosine decay: from lr_max to lr_ms2_gamma * lr_max until lr_ms2 fraction.
    4. Cosine decay: from lr_ms2_gamma * lr_max to lr_min until lr_ms3 fraction (if lr_ms3 provided), else until end.
    5. (Optional) Constant: hold at lr_min for remainder if lr_ms3 provided.
    """
    def __init__(self, optimizer, config, total_mini_epochs, last_epoch=-1):
        max_lr = config.lr_max
        min_lr = config.lr_min or 1e-6
        wu_frac = getattr(config, "lr_wu", 0.0)
        wu_gamma = getattr(config, "lr_wu_gamma", 0.04)
        ms1_frac = config.lr_ms1
        ms2_frac = config.lr_ms2
        ms3_frac = getattr(config, "lr_ms3", None)  # Optional extra milestone
        ms2_gamma = getattr(config, "lr_ms2_gamma", 0.1)

        if isinstance(max_lr, (list, tuple)):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.warmup_steps = int(wu_frac * total_mini_epochs)
        self.hold_steps = int(ms1_frac * total_mini_epochs) - self.warmup_steps
        self.decay1_steps = int(ms2_frac * total_mini_epochs) - (self.warmup_steps + self.hold_steps)
        if self.decay1_steps < 0:
            self.decay1_steps = 0

        if ms3_frac is not None:
            self.decay2_steps = int(ms3_frac * total_mini_epochs) - (self.warmup_steps + self.hold_steps + self.decay1_steps)
            if self.decay2_steps < 0:
                self.decay2_steps = 0
            used = self.warmup_steps + self.hold_steps + self.decay1_steps + self.decay2_steps
            self.final_steps = total_mini_epochs - used
            if self.final_steps < 0:
                self.final_steps = 0
        else:
            # No ms3 milestone: second cosine covers remainder; no constant phase
            self.decay2_steps = total_mini_epochs - (self.warmup_steps + self.hold_steps + self.decay1_steps)
            if self.decay2_steps < 0:
                self.decay2_steps = 0
            self.final_steps = 0

        self.total_steps = total_mini_epochs
        self.wu_gamma = wu_gamma
        self.ms2_gamma = ms2_gamma
        self.ms3_frac = ms3_frac

        self.optimizer = optimizer
        self.base_lrs = self.max_lrs.copy()
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        wu, hold, d1, d2, fin = self.warmup_steps, self.hold_steps, self.decay1_steps, self.decay2_steps, self.final_steps

        # Phase 1: Warmup
        if t < wu:
            return [
                (self.wu_gamma * max_lr) + (max_lr - self.wu_gamma * max_lr) * (float(t + 1) / max(1, wu))
                for max_lr in self.max_lrs
            ]

        t_adj = t - wu
        # Phase 2: Hold
        if t_adj < hold:
            return self.max_lrs

        t_adj -= hold
        # Phase 3: First cosine decay (max_lr -> ms2_gamma * max_lr)
        if t_adj < d1:
            return [
                (self.ms2_gamma * max_lr) + 0.5 * (max_lr - self.ms2_gamma * max_lr) *
                (1 + math.cos(math.pi * t_adj / max(1, d1)))
                for max_lr in self.max_lrs
            ]

        t_adj -= d1
        # Phase 4: Second cosine decay (ms2_gamma * max_lr -> min_lr)
        if t_adj < d2:
            return [
                min_lr + 0.5 * (self.ms2_gamma * max_lr - min_lr) *
                (1 + math.cos(math.pi * t_adj / max(1, d2)))
                for max_lr, min_lr in zip(self.max_lrs, self.min_lrs)
            ]

        # Phase 5: Constant at min_lr (only if ms3 provided)
        return self.min_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lrs = self.get_lr()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg['lr'] = lr

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            'warmup_steps': self.warmup_steps,
            'hold_steps': self.hold_steps,
            'decay1_steps': self.decay1_steps,
            'decay2_steps': self.decay2_steps,
            'final_steps': self.final_steps,
            'max_lrs': self.max_lrs,
            'min_lrs': self.min_lrs,
            'wu_gamma': self.wu_gamma,
            'ms2_gamma': self.ms2_gamma,
            'ms3_frac': self.ms3_frac,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.warmup_steps = state_dict['warmup_steps']
        self.hold_steps = state_dict['hold_steps']
        self.decay1_steps = state_dict['decay1_steps']
        self.decay2_steps = state_dict['decay2_steps']
        self.final_steps = state_dict.get('final_steps', 0)
        self.max_lrs = state_dict['max_lrs']
        self.min_lrs = state_dict['min_lrs']
        self.wu_gamma = state_dict['wu_gamma']
        self.ms2_gamma = state_dict['ms2_gamma']
        self.ms3_frac = state_dict.get('ms3_frac', None)
