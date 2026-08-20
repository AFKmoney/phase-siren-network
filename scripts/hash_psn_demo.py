"""Hash-PSN demo: build weights from data via hashing, generate text.

This is the direct approach: hash(tokens) -> distribute into expert weights.
No training loop. Data goes in, weights get populated, model generates.
"""
import sys, os, json, time
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psn.training.shakespeare import ShakespeareData
from psn.core.hash_distributor import HashBasedPSN


def main():
    print("=" * 70)
    print("  HASH-PSN: Direct Data-to-Weight Construction")
    print("  No gradient. No training loop. Hash -> expert weights.")
    print("=" * 70)
    t0 = time.time()

    # Load Shakespeare
    data = ShakespeareData(seq_len=64)
    token_ids = np.array([data.char_to_idx.get(ch, 0) for ch in data.text], dtype=np.int32)
    print(f"Data: {len(token_ids)} tokens, vocab={data.vocab_size} [{time.time()-t0:.1f}s]")

    # Build model from data via hashing
    model = HashBasedPSN(
        num_experts=32,
        weight_dim=64,
        hash_dim=64,
        kuramoto_K=2.0,
        vocab_size=data.vocab_size,
        seed=42,
    )

    model.build_from_data(token_ids, data.vocab_size, passes=3)
    print(f"Build done [{time.time()-t0:.1f}s]")

    # Generate from multiple seeds
    print(f"\n{'='*70}")
    print("  GENERATION")
    print("=" * 70)

    seeds = ["HAMLET ", "To be, ", "ROMEO: ", "All the ", "If music ",
            "Now is ", "The slings ", "O spirit ", "Soft you ", "Thus con "]
    samples = []
    all_gen = ""

    for seed in seeds:
        gen_text = model.generate(
            seed, data.encode, data.decode,
            length=200, temperature=0.8, use_kuramoto=True,
        )
        samples.append({"seed": seed, "text": gen_text})
        all_gen += gen_text
        print(f"\n  [{seed.strip()}]")
        print(f"  {gen_text[:150]}")

    # Temperature sweep
    print(f"\n{'='*70}")
    print("  TEMPERATURE SWEEP (HAMLET)")
    print("=" * 70)
    temp_samples = {}
    for temp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        gen = model.generate(
            "HAMLET ", data.encode, data.decode,
            length=150, temperature=temp, use_kuramoto=True,
        )
        temp_samples[f"T={temp}"] = gen[:100]
        print(f"  T={temp}: {gen[:80]}")

    # Phase info
    phase_info = model.get_phase_info()
    print(f"\n{'='*70}")
    print("  KURAMOTO STATE")
    print("=" * 70)
    print(f"  Order parameter: {phase_info['order_parameter']:.4f}")
    print(f"  Phase spread: {phase_info['phase_spread']:.3f} rad")

    # Coherence
    print(f"\n{'='*70}")
    print("  COHERENCE")
    print("=" * 70)
    coherence = {}
    for n in [2, 3, 4, 5]:
        ngs = [all_gen[i:i+n] for i in range(len(all_gen) - n + 1)]
        nc = Counter(ngs)
        total = len(ngs)
        unique = len(nc)
        div = unique / total if total > 0 else 0
        mc = nc.most_common(1)[0][1] / total if total > 0 else 0
        coherence[f"{n}gram_diversity"] = round(div, 4)
        coherence[f"{n}gram_max_rep"] = round(mc, 4)
        print(f"  {n}-gram: {unique}/{total} unique ({div:.3f}), max rep: {mc:.3f}")
    cc = Counter(all_gen)
    coherence["vocab_used"] = len(cc)
    print(f"  Vocab: {len(cc)}/{data.vocab_size}")

    al = all_gen.lower()
    words = {}
    for w in ["the", "and", "of", "to", "thou", "thee", "shall", "lord", "love", "heart", "death", "soul", "king", "queen"]:
        c = al.count(f" {w} ") + al.count(f"\n{w} ")
        words[w] = c
    coherence["word_freqs"] = words
    print(f"\n  Shakespeare words:")
    for w, c in sorted(words.items(), key=lambda x: -x[1]):
        print(f"    '{w}': {c}")

    # Expert ownership
    print(f"\n  Expert token ownership:")
    for e in range(min(8, model.num_experts)):
        owned = int((model.distributor.token_freq[e] > 0).sum())
        print(f"    E{e:2d}: {owned} tokens, phase={model.expert_phases[e]:.3f}")

    # Save
    results = {
        "config": {"num_experts": model.num_experts, "weight_dim": model.weight_dim,
                    "vocab_size": data.vocab_size, "total_tokens": len(token_ids)},
        "phase": phase_info, "coherence": coherence,
        "samples": samples, "temperature_sweep": temp_samples,
    }
    out = "download/hash_psn_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> {out}")
    print(f"\n{'='*70}")
    print(f"  TOTAL: {time.time()-t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()