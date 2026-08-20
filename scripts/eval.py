#!/usr/bin/env python3
"""Evaluation protocol for PSN."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psn import PSN, load_corpus

DOMAIN = {
    "shakespeare": ["thou", "thee", "thy", "hath", "lord", "king", "queen", "love", "shall"],
    "bible": ["lord", "god", "unto", "shall", "behold", "saith", "heaven", "israel"],
    "milton": ["heaven", "hell", "adam", "eve", "satan", "paradise", "angel", "sin"],
    "stoic": ["reason", "virtue", "nature", "soul", "mind", "fortune", "wise"],
}


def ngram_stats(text: str):
    out = {}
    for n in (2, 3, 4):
        grams = [text[i : i + n] for i in range(len(text) - n + 1)]
        if not grams:
            continue
        c = Counter(grams)
        total = len(grams)
        out[f"{n}gram_div"] = round(len(c) / total, 4)
        out[f"{n}gram_maxrep"] = round(c.most_common(1)[0][1] / total, 4)
    return out


def domain_recall(text: str):
    low = text.lower()
    hits = {}
    total = 0
    for name, words in DOMAIN.items():
        n = sum(low.count(f" {w} ") for w in words)
        hits[name] = n
        total += n
    hits["total"] = total
    return hits


def main():
    print("=" * 64)
    print("  PSN eval")
    print("=" * 64)

    token_ids, c2i, i2c, vocab = load_corpus()
    encode = lambda s: [c2i.get(c, 0) for c in s]
    decode = lambda ids: "".join(i2c.get(i, "") for i in ids)

    t0 = time.time()
    model = PSN(num_experts=64, weight_dim=128, vocab_size=vocab)
    model.build(token_ids, vocab, passes=2, primary_mass=0.92)
    print(f"Build {time.time()-t0:.1f}s")

    seeds = [
        "HAMLET ",
        "To be, or not to be",
        "In the beginning God",
        "Of Mans First Disobedience",
        "The Lord is my shepherd",
        "Reason and virtue",
    ]
    temps = [0.40, 0.48, 0.55]

    all_text = ""
    samples = []
    for seed in seeds:
        for temp in temps:
            gen = model.generate(
                seed, encode, decode, length=160, temperature=temp, expert_strength=0.80
            )
            samples.append({"seed": seed, "temp": temp, "text": gen})
            all_text += gen + "\n"

    report = {
        "ngram": ngram_stats(all_text),
        "domain_recall": domain_recall(all_text),
        "the_rate": round(all_text.lower().count(" the ") / max(len(all_text.split()), 1), 4),
        "expert_util_mean": float(np.mean(model.distributor.expert_token_count)),
        "expert_util_std": float(np.std(model.distributor.expert_token_count)),
        "kuramoto_r": model.phase_info()["order_parameter"],
        "n_samples": len(samples),
    }

    print(json.dumps(report, indent=2))

    print("\n=== samples ===")
    for s in samples[:4]:
        print(f"\n[{s['seed'][:40]}] T={s['temp']}")
        print(s["text"][:180])
        print("-" * 48)

    out = ROOT / "results" / "eval_report.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
