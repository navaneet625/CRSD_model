import os
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

class LLMTokenizer:
    """
    A modern tokenizer wrapper that trains or loads a SentencePiece Unigram tokenizer
    and converts it into a Hugging Face-compatible PreTrainedTokenizerFast.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_file=None):
        self.vocab_file = vocab_file
        self.tokenizer = None
        self.inv = None

        if vocab_file and os.path.exists(vocab_file):
            print(f"🔠 Loading pre-trained tokenizer from: {vocab_file}")

            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=vocab_file,
                pad_token=self.PAD_TOKEN,
                unk_token=self.UNK_TOKEN,
                bos_token=self.BOS_TOKEN,
                eos_token=self.EOS_TOKEN,
            )

            # Safety: ensure all IDs exist
            self._ensure_special_ids()

            vocab = self.tokenizer.get_vocab()
            self.inv = {v: k for k, v in vocab.items()}

        else:
            print("⚠️ Tokenizer not loaded. Must be trained first.")

    def _ensure_special_ids(self):
        """Ensures the tokenizer has all required special token IDs."""
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": self.PAD_TOKEN})
        if self.tokenizer.unk_token_id is None:
            print("⚠️ Adding missing UNK token ID...")
            self.tokenizer.add_special_tokens({"unk_token": self.UNK_TOKEN})
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.add_special_tokens({"bos_token": self.BOS_TOKEN})
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": self.EOS_TOKEN})

    def train_and_save(self, files, vocab_size=50000, save_path="llm_tokenizer.json"):
        """
        Trains a new SentencePiece Unigram tokenizer and wraps it
        into a PreTrainedTokenizerFast for direct Hugging Face use.
        """
        print(f"⚙️ Training new SentencePiece tokenizer with vocab size: {vocab_size}")

        tokenizer = SentencePieceUnigramTokenizer()

        tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
        )

        # ✅ Correct: add special tokens as a list, not dict
        tokenizer.add_special_tokens(self.SPECIAL_TOKENS)

        # Save the tokenizer
        tokenizer.save(save_path)
        print(f"✅ Tokenizer saved to: {save_path}")

        # Reload with PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=save_path,
            pad_token=self.PAD_TOKEN,
            unk_token=self.UNK_TOKEN,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
        )

        self._ensure_special_ids()

        # Build inverse vocab
        self.inv = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.vocab_file = save_path

        print("📘 Reloaded tokenizer and verified special token IDs.")

    def encode(self, text, add_special_tokens=True):
        """Encodes text into token IDs."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load first.")
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids):
        """Decodes token IDs back to text."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load first.")
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return self.tokenizer.vocab_size if self.tokenizer else 0
