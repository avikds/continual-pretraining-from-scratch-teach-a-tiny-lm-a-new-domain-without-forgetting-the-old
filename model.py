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

