import os
import re
from .tokenizer import LLMTokenizer

TOKENIZER_VOCAB_FILE = "llm_tokenizer.json"

SENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog barks at the quick fox, but a fast brown animal leaps above a sleepy canine.",
    "An LLM is a powerful tool for research purposes.",
]

# ============================================================
# ✅ Minimal built-in Character-level Tokenizer
# ============================================================
class CharTokenizer:
    """
    Simple character-level tokenizer.
    Maps each unique character to an integer ID.
    """
    def __init__(self, text_file):
        with open(text_file, "rb") as f:
            data = f.read()
        chars = [bytes([i]) for i in range(256)]
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inv = {i: ch for ch, i in self.vocab.items()}

        # Add special tokens
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_id = len(self.vocab)
        self.bos_id = len(self.vocab) + 1
        self.eos_id = len(self.vocab) + 2

        self.vocab[self.pad_token] = self.pad_id
        self.vocab[self.bos_token] = self.bos_id
        self.vocab[self.eos_token] = self.eos_id

        self.inv[self.pad_id] = self.pad_token
        self.inv[self.bos_id] = self.bos_token
        self.inv[self.eos_id] = self.eos_token

    def encode(self, text, add_special_tokens=True):
        ids = [self.vocab.get(ch, self.pad_id) for ch in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return "".join(self.inv.get(i, "") for i in ids)

    def vocab_size(self):
        return len(self.vocab)


# ============================================================
# ✅ Minimal built-in Word-level Tokenizer
# ============================================================
class WordTokenizer:
    """
    Simple word-level tokenizer.
    Splits text using regex (\b\w+\b) and maps each unique word to an ID.
    """
    def __init__(self, text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().lower()

        words = re.findall(r"\b\w+\b", text)
        vocab = sorted(list(set(words)))

        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

        # Add special tokens
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_id = len(vocab)
        self.bos_id = len(vocab) + 1
        self.eos_id = len(vocab) + 2

    def encode(self, text, add_special_tokens=True):
        words = re.findall(r"\b\w+\b", text.lower())
        ids = [self.word2id.get(w, self.pad_id) for w in words]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "") for i in ids)

    def vocab_size(self):
        return len(self.word2id) + 3


# ============================================================
# ✅ Tokenizer Builder (supports "char", "word", "subword")
# ============================================================
def build_tokenizer(dataset_path=None, vocab_size=50000, mode="subword"):
    """
    Build or load tokenizer for CRSD.
    mode = "char", "word", or "subword"
    """

    tokenizer_path = os.path.join(os.path.dirname(__file__), TOKENIZER_VOCAB_FILE)

    # Character-level tokenizer
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

    # Word-level tokenizer
    elif mode == "word":
        if dataset_path and os.path.exists(dataset_path):
            print(f"🪶 Building word-level tokenizer from: {dataset_path}")
            tok = WordTokenizer(dataset_path)
            print(f"📘 Word vocab size: {tok.vocab_size()}")
        else:
            raise FileNotFoundError("Dataset not found for word-level tokenizer.")

    # Subword tokenizer (SentencePiece)
    else:
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

    return tok
