"""Corpus loading helpers for PSN."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_CORPORA = [
    "data/shakespeare.txt",
    "data/bible.txt",
    "data/paradise_lost.txt",
    "data/meditations.txt",
]


def load_corpus(
    paths: List[str] | None = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str], int]:
    """Load and concatenate text files → token_ids, char maps, vocab size."""
    if paths is None:
        paths = DEFAULT_CORPORA

    texts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  skip missing {p}")
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            t = f.read()
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[-1]
            if "*** END OF" in t:
                t = t.split("*** END OF", 1)[0]
        texts.append(t)
        print(f"  {os.path.basename(p):22s} {len(t):>10,}")

    if not texts:
        raise FileNotFoundError("no corpus files found")

    full = "\n\n".join(texts)
    chars = sorted(set(full))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    token_ids = np.array([char_to_idx[c] for c in full], dtype=np.int32)
    print(f"  TOTAL                  {len(token_ids):>10,} tokens | vocab={len(chars)}")
    return token_ids, char_to_idx, idx_to_char, len(chars)
