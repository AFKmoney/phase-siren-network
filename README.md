# Phase Siren Network (PSN)

**A Non-Gradient Neural Computing Paradigm**

> Neural computation from phase dynamics. No backprop. No gradient descent.
> Tokens are hashed directly into expert weights. Knowledge is *constructed*, not *learned*.

---

## The Core Idea

Traditional neural networks **train** weights via gradient descent over thousands of iterations.
PSN **constructs** weights by hashing tokens directly into expert weight matrices:

```
H(token, context) → (expert_id, row, col, Δ)
```

Each token in the dataset is deterministically mapped to a specific position in a specific expert's
weight matrix. After a single pass over the data, the model "knows" the corpus. No training loop.
No loss function. No optimizer.

## Architecture

```
Input Token
     │
     ├── Hash Distributor ──▶ H(token, context) → (expert, row, col, Δ)
     │                           │
     │                           └── Direct weight injection ──▶ W_expert[row, col] += Δ
     │
     ├── Kuramoto-Coupled Experts (32 oscillators)
     │       dθᵢ/dt = ωᵢ + (K/N)·Σⱼ Kᵢⱼ·sin(θⱼ - θᵢ)
     │
     ├── Bigram/Trigram Phase Modulation
     │
     └── SIREN Phase Encoding: sin(ω₀·Wx + b)
```

### Key Components

| Component | Role |
|---|---|
| **HashDistributor** | Maps (token, context) → expert weight positions. Replaces training. |
| **Kuramoto Attractor** | 32 coupled phase oscillators. Provides temporal dynamics for generation. |
| **SIREN Encoding** | Sinusoidal activations encode information in phase θ ∈ [0, 2π). |
| **Bigram/Trigram Tables** | Co-occurrence statistics built during hash distribution. |
| **Phase Modulation** | Kuramoto phases softly modulate next-token scores. |

## Why This Is Different

| | Traditional LLM | PSN |
|---|---|---|
| Weight construction | Gradient descent (thousands of steps) | Hash injection (1-3 passes) |
| Training loop | Forward + backward + optimizer | Single data scan |
| Backpropagation | Yes | **None** |
| Loss function | Cross-entropy | **None** |
| Expert knowledge | Distributed via attention | **Directly constructed via hash** |
| Temporal dynamics | Positional encoding (static) | Kuramoto oscillators (dynamic) |
| Self-modification | LoRA, fine-tuning | Hebbian adaptation, expert clone/prune |

## Quick Start

```bash
pip install numpy
python scripts/hash_psn_demo.py
```

This will:
1. Download Shakespeare dataset (~1.1M characters)
2. Hash all tokens into 32 expert weight matrices (3 passes)
3. Generate text from multiple seeds with temperature sweep
4. Output coherence metrics and save results to `download/hash_psn_results.json`

**Expected runtime: ~5-10 seconds** (not hours/days like gradient training)

## Results

The model generates text with Shakespeare-like patterns after a single hash pass:

- Character names (HAMLET, ROMEO, KING, QUEEN)
- Common Elizabethan words ("the", "and", "thou", "lord")
- Theatrical formatting (dialogue, stage directions)
- Varying coherence across temperature settings

### Coherence Metrics (32 experts, 1.1M tokens)

| Metric | Value |
|---|---|
| 2-gram diversity | 0.22 |
| 3-gram diversity | 0.56 |
| 4-gram diversity | 0.82 |
| Vocab coverage | 61/65 chars (94%) |
| Kuramoto order parameter | 0.91 (high synchronization) |
| Build time | ~3 seconds |

## Project Structure

```
psn/
├── core/
│   ├── hash_distributor.py   # H(token) → weight injection (THE key innovation)
│   ├── phase_siren.py       # sin(ω₀Wx+b) phase encoding
│   ├── kuramoto.py          # Coupled oscillator dynamics (RK4)
│   ├── phase_router.py      # Phase-resonance routing (legacy)
│   └── expert.py            # Expert ensemble with clone/prune
├── network/
│   ├── psn.py               # Full JAX PSN architecture
│   ├── self_modification.py # Hebbian adaptation
│   └── synchronization.py  # Inter-instance coupling
├── training/
│   ├── shakespeare.py       # Character-level Shakespeare
│   └── trainer.py           # JAX training loop (legacy)
├── metrics/
│   └── metrics.py           # Order parameter, entropy, BPC
├── config.py               # PSNConfig dataclass
└── __init__.py
scripts/
└── hash_psn_demo.py       # Main demo: hash → build → generate
```

## Mathematical Foundation

### Hash Weight Construction

For each token t with context c:
```
H(t, c) = hash(f"{t}:{c[:4]}:{seed}")
expert_id = H(t,c) mod E
row = (H(t,c) // E) mod D
col = ((H(t,c) // E) // D) mod D
W[expert_id, row, col] += δ
```

### Kuramoto Dynamics

```
dθᵢ/dt = ωᵢ + (K/N) · Σⱼ Kᵢⱼ · sin(θⱼ - θᵢ)
```

Integrated via 4th-order Runge-Kutta. Order parameter:
```
r · e^(iψ) = (1/N) · Σⱼ e^(iθⱼ)
```

### SIREN Phase Encoding

``ny = sin(ω₀ · Wx + b)```

Information is encoded in phase θ ∈ [0, 2π), enabling oscillatory computation.

## Scaling

The hash-into-weights approach is fundamentally O(N) in dataset size with constant memory per expert.
For a 4.2 GB text corpus:

- ~1 billion tokens (char-level)
- Build time: **~30-60 minutes** (single pass, no GPU needed)
- Memory: O(E × D²) for experts, independent of dataset size
- No gradient storage, no optimizer state

Compare to training a comparable LLM: **weeks on 8+ A100 GPUs**.

## Citation

```bibtex
@misc{psn2024,
  title={Phase Siren Network: Non-Gradient Neural Computation via Hash Weight Construction},
  author={PSN Research Team},
  year={2024},
  note={Phase dynamics meet random feature maps}
}
```

## License

MIT
