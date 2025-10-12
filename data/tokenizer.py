# data/tokenizer.py
import re

class SimpleTokenizer:
    """
    Simple whitespace + punctuation tokenizer for CRSD text data.
    Supports build/encode/decode operations.
    """

    def __init__(self, lowercase=True):
        self.vocab = {}
        self.inv = []
        self.lowercase = lowercase

    def build(self, texts):
        """
        Build vocabulary from a list of sentences or a text corpus.
        """
        token_set = set()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            token_set.update(tokens)

        # add special tokens
        specials = ["<PAD>", "<UNK>"]
        self.inv = specials + sorted(list(token_set))
        self.vocab = {tok: i for i, tok in enumerate(self.inv)}

    def encode(self, text):
        """
        Convert a text string into list of token IDs.
        """
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        unk_id = self.vocab.get("<UNK>", 1)
        return [self.vocab.get(tok, unk_id) for tok in tokens]

    def decode(self, ids):
        """
        Convert list of IDs back to text.
        """
        return " ".join(self.inv[i] for i in ids if i < len(self.inv))
