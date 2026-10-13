"""Byte-level BPE tokenizer from scratch, with training and encoding vectorized in PyTorch.

The corpus is pre-tokenized into words (regex), deduplicated into (word, count) pairs and
flattened into one 1-D int64 tensor with a -1 separator between words. Each merge is then a
handful of tensor ops: count pairs, pick the best, and rewrite the tensor.
"""
import json
import re
from collections import Counter

import torch

# Simplified GPT-2 style pre-tokenization pattern (stdlib `re`, no extra dependency).
PRETOKEN_PATTERN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")
SEP = -1  # word separator inside the flat tensor


def _flatten(words):
    """list[bytes] -> flat tensor of byte ids with SEP between words, and word start offsets."""
    parts, n = [], 0
    for w in words:
        parts.append(torch.frombuffer(bytearray(w), dtype=torch.uint8).to(torch.int64))
        parts.append(torch.tensor([SEP], dtype=torch.int64))
        n += len(w) + 1
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)


def _merge_pair(ids, a, b, new_id, weights=None):
    """Replace every non-overlapping, left-to-right occurrence of (a, b) in `ids` by `new_id`.

    `weights` (optional) is a per-position tensor that is carried along (the weight of the
    left token of each merged pair is kept).
    """
    if ids.numel() < 2:
        return ids, weights
    m = (ids[:-1] == a) & (ids[1:] == b)
    if a == b:
        # Overlapping matches (e.g. "aaa"): keep 1st, 3rd, ... match of every run of matches.
        idx = torch.arange(m.numel(), device=ids.device)
        prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=ids.device), m[:-1]])
        run_start = torch.cummax(torch.where(m & ~prev, idx, torch.full_like(idx, -1)), 0).values
        m = m & ((idx - run_start) % 2 == 0)
    pos = m.nonzero().squeeze(1)
    if pos.numel() == 0:
        return ids, weights
    ids = ids.clone()
    ids[pos] = new_id
    keep = torch.ones_like(ids, dtype=torch.bool)
    keep[pos + 1] = False
    return ids[keep], (weights[keep] if weights is not None else None)


class BPETokenizer:
    def __init__(self):
        self.merges = []  # list[(int, int)] in merge order; merge i creates token 256 + i
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special_tokens = {}  # str -> id
        self._ranks = {}  # (int, int) -> merge rank

    # ------------------------------------------------------------------ training
    def train(self, text, vocab_size, special_tokens=(), verbose=False):
        assert vocab_size >= 256, "vocab_size must be >= 256 (the byte alphabet)"
        counts = Counter(self._pretokenize(text, special_tokens))
        words = list(counts)
        # Per-position weight = frequency of the word the position belongs to.
        ids = _flatten(words)
        lens = torch.tensor([len(w) + 1 for w in words], dtype=torch.int64)
        weights = torch.repeat_interleave(torch.tensor([counts[w] for w in words], dtype=torch.int64), lens)

        self.merges, self.vocab = [], {i: bytes([i]) for i in range(256)}
        n_merges = vocab_size - 256 - len(special_tokens)
        for i in range(max(n_merges, 0)):
            new_id = 256 + i
            valid = (ids[:-1] >= 0) & (ids[1:] >= 0)
            if not valid.any():
                break
            keys = ids[:-1][valid] * new_id + ids[1:][valid]  # new_id bounds all current ids
            uniq, inv = torch.unique(keys, return_inverse=True)
            freq = torch.zeros(uniq.numel(), dtype=torch.int64).scatter_add_(0, inv, weights[:-1][valid])
            best = int(freq.argmax())  # ties -> smallest key, so training is deterministic
            a, b = divmod(int(uniq[best]), new_id)
            ids, weights = _merge_pair(ids, a, b, new_id, weights)
            self.merges.append((a, b))
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            if verbose:
                print(f"merge {i + 1}/{n_merges}: {self.vocab[a]!r} + {self.vocab[b]!r} "
                      f"-> {self.vocab[new_id]!r} (count {int(freq[best])})")
        self._finalize(special_tokens)

    def _finalize(self, special_tokens):
        self._ranks = {p: r for r, p in enumerate(self.merges)}
        self.special_tokens = {}
        for tok in special_tokens:
            tid = len(self.vocab)
            self.special_tokens[tok] = tid
            self.vocab[tid] = tok.encode("utf-8")

    # ------------------------------------------------------------------ encoding
    def _pretokenize(self, text, special_tokens=()):
        """Yield pre-token byte strings; special tokens are split out first."""
        chunks = [text]
        if special_tokens:
            split = re.compile("(" + "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + ")")
            chunks = [c for c in split.split(text) if c]
            chunks = [c for c in chunks if c not in special_tokens]
        for chunk in chunks:
            for m in PRETOKEN_PATTERN.finditer(chunk):
                yield m.group().encode("utf-8")

    def _encode_words(self, words):
        """Apply all merges, in rank order, to a list of unique byte-words at once."""
        ids = _flatten(words)
        if not self.merges:
            return self._split(ids)
        for i, (a, b) in enumerate(self.merges):
            ids, _ = _merge_pair(ids, a, b, 256 + i)
        return self._split(ids)

    @staticmethod
    def _split(ids):
        out, cur = [], []
        for t in ids.tolist():
            if t == SEP:
                out.append(cur)
                cur = []
            else:
                cur.append(t)
        return out

    def encode(self, text):
        """Encode text to a list of token ids. Special tokens in `text` map to their ids."""
        result = []
        if self.special_tokens:
            split = re.compile("(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")")
            chunks = [c for c in split.split(text) if c]
        else:
            chunks = [text]
        cache = {}
        for chunk in chunks:
            if chunk in self.special_tokens:
                result.append(self.special_tokens[chunk])
                continue
            words = [m.group().encode("utf-8") for m in PRETOKEN_PATTERN.finditer(chunk)]
            todo = list(dict.fromkeys(w for w in words if w not in cache))
            if todo:
                cache.update(zip(todo, self._encode_words(todo)))
            for w in words:
                result.extend(cache[w])
        return result

    def encode_tensor(self, text):
        return torch.tensor(self.encode(text), dtype=torch.long)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = [tuple(p) for p in data["merges"]]
        for i, (a, b) in enumerate(tok.merges):
            tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]
        tok._finalize(())
        for t, tid in data["special_tokens"].items():
            tok.special_tokens[t] = tid
            tok.vocab[tid] = t.encode("utf-8")
        return tok

    def __len__(self):
        return len(self.vocab)
