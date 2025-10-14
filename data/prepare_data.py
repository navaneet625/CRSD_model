import os
import re
from .tokenizer import LLMTokenizer

TOKENIZER_VOCAB_FILE = "llm_tokenizer.json"

SENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog barks at the quick fox, but a fast brown animal leaps above a sleepy canine.",
    "An LLM is a powerful tool for research purposes.",
]
class CharTokenizer:
    """
    Byte-level character tokenizer (robust to any Unicode input).
    Each unique byte (0–255) becomes a token ID.
    """
    def __init__(self, text_file):
        with open(text_file, "rb") as f:
            data = f.read()

        # Build byte-level vocab (0–255)
        self.vocab = {bytes([i]): i for i in range(256)}
        self.inv = {i: bytes([i]) for i in range(256)}

        # Add special tokens
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_id = 256
        self.bos_id = 257
        self.eos_id = 258

        self.vocab[self.pad_token] = self.pad_id
        self.vocab[self.bos_token] = self.bos_id
        self.vocab[self.eos_token] = self.eos_id

        self.inv[self.pad_id] = self.pad_token
        self.inv[self.bos_id] = self.bos_token
        self.inv[self.eos_id] = self.eos_token

        print(f"📘 CharTokenizer initialized | vocab_size={self.vocab_size()}")
        print(f"🆔 Special IDs: PAD={self.pad_id}, BOS={self.bos_id}, EOS={self.eos_id}")

    def encode(self, text, add_special_tokens=True):
        """Convert text (any Unicode) to byte IDs safely."""
        text_bytes = text.encode("utf-8", errors="replace")  # Replace unknowns with '?'
        ids = [b for b in text_bytes]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        print(f"🧩 [DEBUG] Encoded {len(ids)} bytes → first 20 tokens: {ids[:20]}")
        return ids

    def decode(self, ids):
        """Convert byte IDs back to readable text."""
        bytes_seq = bytearray()
        for i in ids:
            if i < 256:
                bytes_seq.append(i)
        text = bytes_seq.decode("utf-8", errors="replace")
        print(f"🔡 [DEBUG] Decoded sample → '{text[:100]}'")
        return text

    def vocab_size(self):
        return len(self.vocab)

class WordTokenizer:
    def __init__(self, text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().lower()
        words = re.findall(r"\b\w+\b", text)
        vocab = sorted(list(set(words)))
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_id = len(vocab)
        self.bos_id = len(vocab) + 1
        self.eos_id = len(vocab) + 2

        print(f"📘 WordTokenizer initialized | vocab_size={self.vocab_size()}")
        print(f"🆔 Special IDs: PAD={self.pad_id}, BOS={self.bos_id}, EOS={self.eos_id}")

    def encode(self, text, add_special_tokens=True):
        words = re.findall(r"\b\w+\b", text.lower())
        ids = [self.word2id.get(w, self.pad_id) for w in words]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        print(f"🧩 [DEBUG] Encoding sample: {words[:10]} → {ids[:10]}")
        return ids

    def decode(self, ids):
        text = " ".join(self.id2word.get(i, "") for i in ids)
        print(f"🔡 [DEBUG] Decoding IDs {ids[:10]} → '{text[:80]}'")
        return text

    def vocab_size(self):
        return len(self.word2id) + 3

def build_tokenizer(dataset_path=None, vocab_size=50000, mode="subword"):
    tokenizer_path = os.path.join(os.path.dirname(__file__), TOKENIZER_VOCAB_FILE)

    if mode == "char":
        if dataset_path and os.path.exists(dataset_path):
            print(f"🔡 Building character-level tokenizer from: {dataset_path}")
            tok = CharTokenizer(dataset_path)
        else:
            print("⚙️ Using demo sentences for char-level tokenizer.")
            text = "\n".join(SENTS)
            tmp_file = "/tmp/demo_char.txt"
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(text)
            tok = CharTokenizer(tmp_file)
        print(f"📘 Character vocab size: {tok.vocab_size()}")

    elif mode == "word":
        if dataset_path and os.path.exists(dataset_path):
            print(f"🪶 Building word-level tokenizer from: {dataset_path}")
            tok = WordTokenizer(dataset_path)
            print(f"📘 Word vocab size: {tok.vocab_size()}")
        else:
            raise FileNotFoundError("Dataset not found for word-level tokenizer.")

    else:  # subword
        if os.path.exists(tokenizer_path):
            tok = LLMTokenizer(vocab_file=tokenizer_path)
        elif dataset_path and os.path.exists(dataset_path):
            print(f"⚠️ Tokenizer not found. Training a new one from: {dataset_path}")
            files = [dataset_path]
            tok = LLMTokenizer()
            tok.train_and_save(files=files, vocab_size=vocab_size, save_path=tokenizer_path)
        else:
            raise FileNotFoundError(
                f"Cannot find tokenizer at {tokenizer_path} or dataset {dataset_path}"
            )
        print(f"📘 Loaded subword tokenizer with vocab size: {tok.vocab_size()}")

    # 🔍 Final debug check
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            sample = f.read(300)
        encoded = tok.encode(sample, add_special_tokens=False)
        print(f"🧩 [DEBUG] build_tokenizer sample encode len={len(encoded)}, first 20 tokens={encoded[:20]}")
        decoded = tok.decode(encoded[:100])
        print(f"🔡 [DEBUG] build_tokenizer sample decode='{decoded[:120]}'")

    return tok