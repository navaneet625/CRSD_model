# data/prepare_data.py
import os
from .tokenizer import SimpleTokenizer

# Tiny demo sentences (for fallback / quick tests)
SENTS = [
    "The quick brown fox jumps over the lazy dog",
    "the dog barks at the quick fox",
    "a fast brown animal leaps above a sleepy canine",
]

def build_tokenizer(dataset_path=None):
    """
    Build tokenizer from either:
      1. Provided dataset file path, or
      2. Tiny in-memory example (SENTS)
    """
    tok = SimpleTokenizer()

    if dataset_path and os.path.exists(dataset_path):
        print(f"🔠 Building tokenizer from file: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
        # split roughly into sentences for vocab
        texts = text.split("\n")
        tok.build(texts)
    else:
        print("⚙️ Using demo tokenizer from SENTS")
        tok.build(SENTS)

    return tok
