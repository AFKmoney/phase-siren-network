# Phase Siren Network (PSN)

**Non-gradient neural computation.**  
Tokens are hashed into specialized experts. Weights are *constructed*, not trained.

```
H(token) → primary expert
data pass → direct weight injection
Kuramoto oscillators → temporal modulation
```

No backprop. No loss. No optimizer.

## Quick start

```bash
pip install numpy
python scripts/talk.py
```

```bash
python scripts/eval.py
```

## Layout

```
data/               # corpora (Shakespeare, KJV, Paradise Lost, Meditations)
psn/
  model.py          # HashDistributor + PSN
  data.py           # corpus loader
scripts/
  talk.py           # interactive generation
  eval.py           # metrics protocol
results/            # eval reports
```

## Core idea

1. Every token has a **primary expert** (stable hash of the token alone).
2. A single pass over the corpus injects co-occurrence structure into that expert’s weight matrix.
3. Generation mixes local n-gram statistics with the specialized expert signal (frequency table + SVD projection) and light Kuramoto phase modulation.

## Status

Prototype. Generates coherent local patterns and mixes domain vocabulary (Shakespeare / Bible / Milton / Stoic). Still limited by strong bigram modes (“the the …”).

MIT
