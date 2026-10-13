import os
import tempfile

from bpe import BPETokenizer

CORPUS = ("low low low low low lower lower newest newest newest newest newest newest "
          "widest widest widest aaaa aaa <|endoftext|> héllo wörld 你好 \n\n  end.") * 20


def test_known_merges():
    tok = BPETokenizer()
    tok.train("aaabdaaabac", 259)  # classic example: (a,a) first
    assert tok.merges[0] == (97, 97)
    assert tok.decode(tok.encode("aaabdaaabac")) == "aaabdaaabac"


def test_roundtrip_and_special():
    tok = BPETokenizer()
    tok.train(CORPUS, 320, special_tokens=["<|endoftext|>"])
    for s in ["low lower newest", "aaaa aaa", "héllo wörld 你好", "a<|endoftext|>b", "", "  \n\n x"]:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, s
    assert tok.encode("<|endoftext|>") == [tok.special_tokens["<|endoftext|>"]]
    assert len(tok) <= 320  # may stop early once every word is a single token


def test_compresses():
    tok = BPETokenizer()
    tok.train(CORPUS, 300)
    assert len(tok.encode("newest newest")) < len("newest newest".encode())


def test_save_load():
    tok = BPETokenizer()
    tok.train(CORPUS, 300, special_tokens=["<|endoftext|>"])
    path = os.path.join(tempfile.mkdtemp(), "t.json")
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    s = "newest <|endoftext|> héllo"
    assert tok.encode(s) == tok2.encode(s)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
