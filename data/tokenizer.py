# Very small tokenizer for demo purposes
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv = []

    def build(self, texts):
        s = set()
        for t in texts:
            s.update(t.split())
        self.inv = sorted(list(s))
        self.vocab = {w:i for i,w in enumerate(self.inv)}

    def encode(self, text):
        return [self.vocab[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids)