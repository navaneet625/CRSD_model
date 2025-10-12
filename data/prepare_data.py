# small demo dataset creator
from .tokenizer import SimpleTokenizer

SENTS = [
    "The quick brown fox jumps over the lazy dog",
    "the dog barks at the quick fox",
]

def build_tokenizer():
    tok = SimpleTokenizer()
    tok.build(SENTS)
    return tok

if __name__ == "__main__":
    t = build_tokenizer()
    print("Vocab size:", len(t.inv))