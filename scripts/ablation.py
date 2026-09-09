#!/usr/bin/env python3
"""Ablation harness for PSN's scoring components.

This intentionally does not change psn/model.py. It measures the existing
components independently so we can determine whether gains come from n-grams,
expert signals, SVD, or Kuramoto modulation.

Run on a corpus with the same character encoding used by PSN.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from psn import PSN


def make_vocab(text: str):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    ids = np.asarray([c2i[c] for c in text], dtype=np.int64)
    return ids, c2i, i2c


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def score_distribution(model: PSN, tokens, pos, mode: str, expert_strength=0.78, use_kuramoto=True):
    cur = int(tokens[pos - 1])
    if pos >= 2:
        key = (int(tokens[pos - 2]), cur)
        if key in model.trigram:
            base = 0.62 * model.trigram[key] + 0.38 * model.bigram_probs[cur]
        else:
            base = model.bigram_probs[cur].copy()
    else:
        base = model.bigram_probs[min(cur, model.vocab_size - 1)].copy()

    if mode == "ngram":
        return base

    e = model.distributor.get_primary(cur)
    freq = model.distributor.token_freq[e]
    freq = freq / (freq.max() + 1e-8) if freq.max() > 0 else np.ones(model.vocab_size) * 0.01
    head = model.expert_heads[e]

    ctx = tokens[max(0, pos - 4):pos]
    ctx_emb = model.embedding[cur].copy()
    for c in ctx[-3:]:
        ctx_emb += model.embedding[min(int(c), model.vocab_size - 1)]
    ctx_emb /= 4.0
    proj = ctx_emb @ model.expert_proj[e]
    proj -= proj.min()
    if proj.max() > 1e-8:
        proj /= proj.max()

    if mode == "freq":
        signal = freq
    elif mode == "head":
        signal = head
    elif mode == "proj":
        signal = proj
    else:
        signal = 0.35 * freq + 0.20 * head + 0.45 * proj

    scores = base.copy()
    top = np.argsort(-scores)[:22]
    for i in top:
        scores[i] *= 1.0 + expert_strength * signal[i]

    if mode == "full" and use_kuramoto:
        phase = model.expert_phases[e]
        for i in top[:10]:
            tp = 2 * np.pi * i / max(model.vocab_size, 1)
            scores[i] *= 1.0 + 0.07 * np.cos(phase - tp)
    return scores


def evaluate(model, test_tokens, modes, limit):
    rows = []
    n = min(len(test_tokens), limit + 4)
    for mode in modes:
        nll = 0.0
        correct = 0
        ent = 0.0
        repeats = 0
        count = 0
        for pos in range(4, n - 1):
            scores = np.maximum(score_distribution(model, test_tokens, pos, mode), 1e-12)
            p = scores / scores.sum()
            target = int(test_tokens[pos])
            nll -= math.log(float(p[target]))
            correct += int(int(np.argmax(p)) == target)
            ent += entropy(p)
            repeats += int(target in test_tokens[max(0, pos - 6):pos])
            count += 1
        rows.append({
            "mode": mode,
            "tokens": count,
            "cross_entropy_nats": nll / max(count, 1),
            "perplexity": math.exp(nll / max(count, 1)),
            "top1_accuracy": correct / max(count, 1),
            "mean_entropy_bits": ent / max(count, 1),
            "recent_repeat_rate": repeats / max(count, 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--test-chars", type=int, default=50000)
    ap.add_argument("--build-chars", type=int, default=500000)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("results/ablation.json"))
    args = ap.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    if len(text) < args.test_chars + args.build_chars + 8:
        raise SystemExit("Corpus is too small for the requested build/test split.")

    # Disjoint contiguous regions: construction never sees evaluation chars.
    train_text = text[:args.build_chars]
    test_text = text[args.build_chars:args.build_chars + args.test_chars]
    combined = train_text + test_text
    train_ids, c2i, _ = make_vocab(combined)
    split = len(train_text)
    train_ids = train_ids[:split]
    test_ids = train_ids.__class__([c2i[c] for c in test_text])
    V = len(c2i)

    model = PSN(num_experts=args.experts, weight_dim=args.dim, vocab_size=V, seed=args.seed)
    model.build(train_ids, V, passes=args.passes)

    modes = ["ngram", "freq", "head", "proj", "expert", "full"]
    rows = evaluate(model, test_ids, modes, args.test_chars - 4)
    result = {
        "config": vars(args) | {"vocab_size": V},
        "split": {"train_chars": len(train_text), "test_chars": len(test_text)},
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
