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

