"""
Continual Pretraining from Scratch: Teach a Tiny LM a New Domain Without Forgetting the Old

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fetch_text
import os
import tempfile
import urllib.request

URLS = {
    "stories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def fetch_text(url, name, max_chars):
    # Cache the corpus in the system's temporary directory.
    path = os.path.join(tempfile.gettempdir(), f"cpt_{name}.txt")

    # Download the corpus only if it is not already cached.
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    # Read the first max_chars characters using UTF-8 encoding.
    # Ignore invalid UTF-8 sequences to prevent decoding errors.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_chars)

    return text

# Step 2 - CharTokenizer
import torch

class CharTokenizer:
    def __init__(self, texts):
        # Build one shared vocabulary from all supplied text domains.
        self.itos = sorted(set("".join(texts)))

        # Create the inverse mapping from characters to integer IDs.
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, s):
        # Convert characters to IDs, silently skipping unknown characters.
        # Return a one-dimensional torch.long tensor.
        return torch.tensor(
            [self.stoi[ch] for ch in s if ch in self.stoi],
            dtype=torch.long
        )

    def decode(self, ids):
        # Support both Python lists and one-dimensional PyTorch tensors.
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Convert integer IDs back into their corresponding characters.
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self):
        # Return the number of unique characters in the vocabulary.
        return len(self.itos)

# Step 3 - make_batch
def make_batch(ids, T, batch, generator):
    # Sample starting positions uniformly using the supplied generator.
    # The upper bound is exclusive, ensuring every target window fits.
    starts = torch.randint(
        0,
        len(ids) - T,
        (batch,),
        generator=generator
    )

    # Construct input sequences of length T.
    x = torch.stack([
        ids[s : s + T] for s in starts.tolist()
    ])

    # Construct next-character targets by shifting each input by one.
    y = torch.stack([
        ids[s + 1 : s + T + 1] for s in starts.tolist()
    ])

    return x, y

# Step 4 - load_domains
def load_domains(max_chars=600000, val_frac=0.1):
    # Fetch both text corpora using the existing cached downloader.
    stories = fetch_text(URLS["stories"], "stories", max_chars)
    shakespeare = fetch_text(URLS["shakespeare"], "shakespeare", max_chars)

    # Build one shared character vocabulary across both domains.
    tok = CharTokenizer([stories, shakespeare])

    # Encode both domains using the same tokenizer.
    stories_ids = tok.encode(stories)
    shakespeare_ids = tok.encode(shakespeare)

    # Split each token stream by position to prevent validation text
    # from appearing in training windows.
    stories_split = int((1 - val_frac) * len(stories_ids))
    shakespeare_split = int((1 - val_frac) * len(shakespeare_ids))

    # Package both domains using the required structure.
    return {
        "tok": tok,
        "A": {
            "name": "stories",
            "train": stories_ids[:stories_split],
            "val": stories_ids[stories_split:],
        },
        "B": {
            "name": "shakespeare",
            "train": shakespeare_ids[:shakespeare_split],
            "val": shakespeare_ids[shakespeare_split:],
        },
    }

# Step 5 - TinyGPT
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()

        # Pre-normalization layers.
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Multi-head self-attention projections.
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

        # Position-wise feed-forward network.
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

        self.heads = heads

    def forward(self, x):
        # Input shape: (B, T, d)
        B, T, C = x.shape

        # Normalize and project the input into queries, keys, and values.
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        # Split the embedding dimension into multiple attention heads.
        head_dim = C // self.heads

        q = q.view(B, T, self.heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, head_dim).transpose(1, 2)

        # Causal attention prevents each position from seeing future tokens.
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # Merge the attention heads back into the embedding dimension.
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # First residual connection: self-attention.
        x = x + self.proj(attn)

        # Second residual connection: feed-forward network.
        x = x + self.fc2(
            F.gelu(self.fc1(self.ln2(x)))
        )

        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=64, heads=4, layers=2, max_len=64):
        super().__init__()

        # Character-token and positional embeddings.
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(max_len, d)

        # Stack the requested number of transformer blocks.
        self.blocks = nn.ModuleList([
            Block(d, heads) for _ in range(layers)
        ])

        # Final normalization and vocabulary projection.
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

        self.max_len = max_len

    def forward(self, idx):
        # idx has shape (B, T).
        B, T = idx.shape

        # Construct position IDs on the same device as the input.
        pos = torch.arange(T, device=idx.device)

        # Add token and positional embeddings.
        x = self.wte(idx) + self.wpe(pos)

        # Apply the causal transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Produce next-character logits of shape (B, T, vocab).
        x = self.lnf(x)
        logits = self.head(x)

        return logits

# Step 6 - lm_loss
def lm_loss(model, x, y):
    # Forward pass: obtain logits of shape (B, T, vocab).
    logits = model(x)

    # Get the vocabulary size from the model's output layer.
    vocab = model.head.out_features

    # Flatten logits to (B * T, vocab) and targets to (B * T).
    logits = logits.reshape(-1, vocab)
    targets = y.reshape(-1)

    # Compute mean cross-entropy loss in nats per character.
    return F.cross_entropy(logits, targets)

# Step 7 - eval_loss
@torch.no_grad()
def eval_loss(model, ids, T, n_batches, seed, batch=32):
    # Create a dedicated random generator for reproducible evaluation.
    generator = torch.Generator().manual_seed(seed)

    total_loss = 0.0

    # Evaluate on the same sequence windows whenever the seed is unchanged.
    for _ in range(n_batches):
        x, y = make_batch(ids, T, batch, generator)

        # Compute the mean next-character cross-entropy for this batch.
        loss = lm_loss(model, x, y)
        total_loss += loss.item()

    # Return the mean validation loss as a Python float.
    return total_loss / n_batches

# Step 8 - cosine_lr
import math


def cosine_lr(step, total, warmup, lr_max, lr_min):
    # Linear warm-up from zero to the maximum learning rate.
    if step < warmup:
        return lr_max * step / warmup

    # Compute progress through the cosine-decay phase.
    # Protect against division by zero for short schedules.
    p = (step - warmup) / max(total - 1 - warmup, 1)

    # Keep progress within [0, 1].
    p = max(0.0, min(1.0, p))

    # Cosine decay from lr_max to lr_min.
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1.0 + math.cos(math.pi * p)
    )

# Step 9 - train_phase
def window_sampler(ids, T):
    # Return a batch function that samples windows of length T
    # from the given token stream using the supplied generator.
    def batch_fn(generator, batch):
        return make_batch(ids, T, batch, generator)

    return batch_fn


def train_phase(
    model,
    steps,
    cfg,
    seed,
    batch_fn,
    extra_loss=None,
    monitor=None,
    log_every=25,
    params=None,
):
    # Use a dedicated random generator for reproducible batch sampling.
    generator = torch.Generator().manual_seed(seed)

    # Train all model parameters unless a specific parameter collection
    # is provided, for example, trainable LoRA parameters.
    params = list(model.parameters()) if params is None else list(params)

    # Initialize AdamW with zero weight decay as required.
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["lr_max"],
        weight_decay=0.0,
    )

    history = []

    for s in range(steps):
        # Update the learning rate for every optimizer parameter group.
        lr = cosine_lr(
            s,
            steps,
            cfg["warmup"],
            cfg["lr_max"],
            cfg["lr_min"],
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # Sample a batch of inputs and next-character targets.
        x, y = batch_fn(generator, cfg["batch"])

        # Compute the data loss separately so the history excludes
        # any optional regularization or auxiliary loss.
        data_loss = lm_loss(model, x, y)

        # Add an optional model-dependent loss, such as EWC.
        if extra_loss is not None:
            loss = data_loss + extra_loss(model)
        else:
            loss = data_loss

        # Perform the optimization step.
        optimizer.zero_grad()
        loss.backward()

        # Clip the gradient norm to stabilize training.
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        # Record the data loss, learning rate, and current step.
        record = {
            "step": s,
            "lr": lr,
            "loss": float(data_loss.detach().item()),
        }

        # Optionally track the original-domain validation loss.
        if (
            monitor is not None
            and (s % log_every == 0 or s == steps - 1)
        ):
            record["old"] = eval_loss(
                model,
                monitor["ids"],
                cfg["T"],
                monitor["n_batches"],
                monitor["seed"],
            )

        history.append(record)

    return history

# Step 10 - domain_losses
def domain_losses(model, domains, T=64, n_batches=16, seed=7):
    # Evaluate the model on domain A's validation split.
    loss_A = eval_loss(
        model,
        domains["A"]["val"],
        T,
        n_batches,
        seed,
    )

    # Evaluate the model on domain B's validation split using
    # the same seed and evaluation settings for a fair comparison.
    loss_B = eval_loss(
        model,
        domains["B"]["val"],
        T,
        n_batches,
        seed,
    )

    # Return validation losses keyed by domain.
    return {
        "A": loss_A,
        "B": loss_B,
    }

# Step 11 - forgetting_report
def forgetting_report(before, after, decimals=4):
    # Record the original-domain validation losses before and after training.
    A_before = before["A"]
    A_after = after["A"]

    # Record the new-domain validation losses before and after training.
    B_before = before["B"]
    B_after = after["B"]

    # Positive forgetting means the original domain's loss increased.
    forgetting = A_after - A_before

    # Positive acquisition means the new domain's loss decreased.
    acquisition = B_before - B_after

    # Return all metrics rounded to the requested precision as Python floats.
    return {
        "A_before": round(float(A_before), decimals),
        "A_after": round(float(A_after), decimals),
        "B_before": round(float(B_before), decimals),
        "B_after": round(float(B_after), decimals),
        "forgetting": round(float(forgetting), decimals),
        "acquisition": round(float(acquisition), decimals),
    }

# Step 12 - old_loss_curve
def old_loss_curve(history, decimals=4):
    # Keep only training records that include monitored old-domain loss.
    records = [record for record in history if "old" in record]

    # Handle phases where no validation loss was monitored.
    if not records:
        return {
            "steps": [],
            "losses": [],
            "peak": None,
            "peak_step": None,
            "final": None,
            "rise": None,
        }

    # Extract the original-domain loss values and corresponding steps.
    steps = [record["step"] for record in records]
    raw_losses = [float(record["old"]) for record in records]

    # Find the first occurrence of the maximum old-domain loss.
    peak_index = raw_losses.index(max(raw_losses))
    peak = raw_losses[peak_index]

    # Measure how far the loss rose above its initial monitored value.
    rise = peak - raw_losses[0]

    # Return the monitored loss curve and its summary statistics.
    return {
        "steps": steps,
        "losses": [round(loss, decimals) for loss in raw_losses],
        "peak": round(peak, decimals),
        "peak_step": steps[peak_index],
        "final": round(raw_losses[-1], decimals),
        "rise": round(rise, decimals),
    }

# Step 13 - replay_batch
def replay_batch(new_ids, old_ids, T, batch, frac, generator):
    # Determine how many windows to replay from the old domain.
    n_old = int(round(frac * batch))
    n_new = batch - n_old

    # Sample new-domain windows first, using the shared generator.
    if n_new > 0:
        x_new, y_new = make_batch(new_ids, T, n_new, generator)

    # Sample old-domain replay windows after the new-domain windows.
    if n_old > 0:
        x_old, y_old = make_batch(old_ids, T, n_old, generator)

    # Handle frac=1: the batch contains only old-domain windows.
    if n_new == 0:
        return x_old, y_old

    # Handle frac=0: the batch contains only new-domain windows.
    if n_old == 0:
        return x_new, y_new

    # Concatenate along the batch dimension, keeping new rows first.
    x = torch.cat((x_new, x_old), dim=0)
    y = torch.cat((y_new, y_old), dim=0)

    return x, y


def replay_sampler(new_ids, old_ids, T, frac):
    # Return a sampler compatible with train_phase.
    def batch_fn(generator, batch):
        return replay_batch(
            new_ids,
            old_ids,
            T,
            batch,
            frac,
            generator,
        )

    return batch_fn

# Step 14 - diagonal_fisher
def diagonal_fisher(model, ids, T, n_batches, seed, batch=8):
    # Use a dedicated generator for reproducible batch sampling.
    generator = torch.Generator().manual_seed(seed)

    # Initialize a diagonal Fisher accumulator for every named parameter.
    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }

    # Accumulate elementwise squared gradients across batches.
    for _ in range(n_batches):
        x, y = make_batch(ids, T, batch, generator)

        # Clear gradients before calculating the current batch's gradients.
        model.zero_grad(set_to_none=False)

        # Compute the next-character loss and its gradients.
        loss = lm_loss(model, x, y)
        loss.backward()

        # Add squared gradients for each parameter.
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name].add_(param.grad.detach().square())

    # Average the accumulated squared gradients across batches.
    if n_batches > 0:
        for name in fisher:
            fisher[name].div_(n_batches)

    # Leave the model's gradients cleared after the estimation.
    model.zero_grad(set_to_none=False)

    return fisher


def fisher_summary(fisher, k=3):
    # Compute the mean Fisher value for each parameter tensor.
    summary = [
        (name, float(values.mean().item()))
        for name, values in fisher.items()
    ]

    # Sort by importance, from largest mean Fisher value to smallest.
    summary.sort(key=lambda item: item[1], reverse=True)

    # Return the top-k parameter tensors and their mean Fisher values.
    return summary[:k]

# Step 15 - ewc_penalty
def ewc_penalty(model, fisher, lam):
    # Snapshot the current parameters as the reference point (theta*).
    # Clone and detach so subsequent training cannot modify the anchor.
    theta_star = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    def penalty(m):
        # Accumulate the Fisher-weighted squared distance from the anchor.
        terms = []

        for name, p in m.named_parameters():
            if name in fisher:
                # Keep the anchor and Fisher tensor compatible with the
                # current parameter's device and data type.
                anchor = theta_star[name].to(
                    device=p.device,
                    dtype=p.dtype,
                )
                importance = fisher[name].to(
                    device=p.device,
                    dtype=p.dtype,
                )

                terms.append(
                    (importance * (p - anchor).square()).sum()
                )

        # EWC regularization: (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2
        if terms:
            return (lam / 2.0) * torch.stack(terms).sum()

        # Return a scalar tensor if no parameters match the Fisher dictionary.
        return next(m.parameters()).sum() * 0.0

    return penalty

# Step 16 - LoRALinear
class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()

        # Store the original linear layer without modifying its parameters.
        self.base = base

        # Low-rank adapter matrices:
        # A projects input features into a lower-dimensional space.
        self.A = nn.Parameter(
            torch.randn(rank, base.in_features) * 0.01
        )

        # B projects the low-rank representation back to output features.
        # Zero initialization ensures the adapter initially contributes zero.
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank)
        )

        # Scale the adapter update by alpha / rank.
        self.scale = alpha / rank

        # Allow the adapter contribution to be enabled or disabled.
        self.enabled = True

    def forward(self, x):
        # When disabled, return the original layer's output exactly.
        if not self.enabled:
            return self.base(x)

        # Compute the low-rank update: x @ A.T @ B.T.
        adapter = x @ self.A.T @ self.B.T

        # Add the scaled adapter update to the original output.
        return self.base(x) + self.scale * adapter

# Step 17 - add_lora
def add_lora(model, rank, alpha):
    # Attach a LoRA adapter to each specified projection in every block.
    projections = ("qkv", "proj", "fc1", "fc2")

    for block in model.blocks:
        for name in projections:
            # Retrieve the existing linear layer.
            base = getattr(block, name)

            # Wrap it only if it has not already been wrapped.
            if not isinstance(base, LoRALinear):
                setattr(
                    block,
                    name,
                    LoRALinear(base, rank=rank, alpha=alpha),
                )

    # Freeze the base model and all non-adapter parameters.
    # Only the LoRA A and B matrices remain trainable.
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith(".A") or name.endswith(".B"))

    # Return the parameters that the optimizer should update.
    return [
        param
        for param in model.parameters()
        if param.requires_grad
    ]


def set_lora(model, enabled):
    # Enable or disable every LoRA adapter in the model.
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = bool(enabled)


def lora_param_count(model):
    # Count the elements in all LoRA adapter matrices.
    return sum(
        param.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for param in (module.A, module.B)
    )

