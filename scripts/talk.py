#!/usr/bin/env python3
"""Interactive talk mode for PSN."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psn import PSN, load_corpus


def main():
    print("=" * 64)
    print("  PSN talk — specialized experts, multi-source")
    print("=" * 64)

    token_ids, c2i, i2c, vocab = load_corpus()
    encode = lambda s: [c2i.get(c, 0) for c in s]
    decode = lambda ids: "".join(i2c.get(i, "") for i in ids)

    print("\nBuilding (64 experts, dim=128)...")
    t0 = time.time()
    model = PSN(num_experts=64, weight_dim=128, vocab_size=vocab, kuramoto_K=2.4)
    model.build(token_ids, vocab, passes=2, primary_mass=0.92)
    print(f"Build {time.time() - t0:.1f}s\n")
    model.specialization_report(i2c)

    print("\n" + "=" * 64)
    print("  Type a seed. Commands: /temp 0.4  /str 0.8  /len 300  /quit")
    print("=" * 64)

    temp, strength, length = 0.48, 0.78, 280

    while True:
        try:
            seed = input("\nseed > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not seed:
            seed = "HAMLET "
        low = seed.lower()
        if low in ("/q", "/quit", "quit", "exit"):
            print("bye.")
            break
        if seed.startswith("/temp "):
            try:
                temp = float(seed.split()[1])
                print(f"  temperature = {temp}")
            except Exception:
                print("  usage: /temp 0.45")
            continue
        if seed.startswith("/str "):
            try:
                strength = float(seed.split()[1])
                print(f"  expert_strength = {strength}")
            except Exception:
                print("  usage: /str 0.75")
            continue
        if seed.startswith("/len "):
            try:
                length = int(seed.split()[1])
                print(f"  length = {length}")
            except Exception:
                print("  usage: /len 300")
            continue

        t0 = time.time()
        out = model.generate(
            seed, encode, decode,
            length=length, temperature=temp, expert_strength=strength,
        )
        print(f"\n{out}\n")
        print(f"  [{time.time()-t0:.2f}s | T={temp} | str={strength}]")


if __name__ == "__main__":
    main()
