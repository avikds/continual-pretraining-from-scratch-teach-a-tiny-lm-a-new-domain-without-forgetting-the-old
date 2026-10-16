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

