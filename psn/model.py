"""Phase Siren Network — hash-constructed specialized experts.

No gradient. No training loop.
Tokens are hashed into primary experts; weights are constructed in one pass.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class HashDistributor:
    """Assign each token a primary expert and inject data into its weights."""

    def __init__(self, num_experts: int, weight_dim: int, seed: int = 42):
        self.num_experts = num_experts
        self.weight_dim = weight_dim
        self.seed = seed

        self.expert_weights = np.zeros(
            (num_experts, weight_dim, weight_dim), dtype=np.float32
        )
        self.expert_biases = np.zeros((num_experts, weight_dim), dtype=np.float32)
        self.token_freq = None  # (E, V)
        self.primary_expert: Dict[int, int] = {}
        self.expert_token_count = np.zeros(num_experts, dtype=np.int64)

        self._p1 = 2654435761
        self._p2 = 2246822519
        self._p3 = 3266489917

    def _primary(self, token_id: int) -> int:
        h = (token_id * self._p1) ^ (self.seed * self._p2)
        return int(h % self.num_experts)

    def distribute(
        self,
        token_ids: np.ndarray,
        vocab_size: int,
        passes: int = 2,
        primary_mass: float = 0.92,
    ) -> Dict:
        N = len(token_ids)
        self.token_freq = np.zeros((self.num_experts, vocab_size), dtype=np.float32)
        token_ids = token_ids.astype(np.int64)

        for t in np.unique(token_ids):
            self.primary_expert[int(t)] = self._primary(int(t))

        print(
            f"[Distributor] {N:,} tokens | {len(self.primary_expert)} unique | "
            f"{self.num_experts} experts | primary_mass={primary_mass}"
        )

        for p in range(passes):
            seed = self.seed + p * 997
            chunk = 250_000
            for start in range(4, N, chunk):
                end = min(start + chunk, N)
                idx = np.arange(start, end)
                toks = token_ids[idx]
                c0 = token_ids[np.maximum(idx - 1, 0)]
                c1 = token_ids[np.maximum(idx - 2, 0)]
                c2 = token_ids[np.maximum(idx - 3, 0)]
                c3 = token_ids[np.maximum(idx - 4, 0)]

                prim = np.array(
                    [self.primary_expert[int(t)] for t in toks], dtype=np.int32
                )

                h = (toks * self._p1) ^ (seed * self._p3)
                h = (h + c0 * self._p2) & 0xFFFFFFFF
                h = (h ^ (c1 * self._p1)) & 0xFFFFFFFF
                h = (h + c2 * self._p3) & 0xFFFFFFFF
                h = (h ^ (c3 * self._p2)) & 0xFFFFFFFF
                rows = ((h // self.num_experts) % self.weight_dim).astype(np.int32)
                cols = (
                    ((h // self.num_experts) // self.weight_dim) % self.weight_dim
                ).astype(np.int32)

                delta = 0.012 * primary_mass
                np.add.at(self.expert_weights, (prim, rows, cols), delta)
                np.add.at(self.expert_biases, (prim, rows), delta * 0.1)
                np.add.at(self.token_freq, (prim, toks), 1.0)

                if primary_mass < 0.99:
                    sec = (prim + (c0 % 3) + 1) % self.num_experts
                    np.add.at(
                        self.expert_weights, (sec, rows, cols), 0.012 * (1 - primary_mass)
                    )
                    np.add.at(self.token_freq, (sec, toks), 0.15)

            print(f"  pass {p + 1}/{passes}")

        for e in range(self.num_experts):
            s = self.token_freq[e].sum()
            if s > 0:
                self.expert_weights[e] /= s ** 0.5
                self.expert_biases[e] /= s ** 0.25
            self.expert_token_count[e] = int((self.token_freq[e] > 0).sum())

        return {
            "expert_weights": self.expert_weights,
            "token_freq": self.token_freq,
            "primary_expert": self.primary_expert,
        }

    def get_primary(self, token_id: int) -> int:
        return self.primary_expert.get(token_id, self._primary(token_id))


class PSN:
    """Hash-constructed specialized expert network + Kuramoto modulation."""

    def __init__(
        self,
        num_experts: int = 64,
        weight_dim: int = 128,
        kuramoto_K: float = 2.4,
        vocab_size: int = 100,
        seed: int = 42,
    ):
        self.num_experts = num_experts
        self.weight_dim = weight_dim
        self.vocab_size = vocab_size
        self.seed = seed

        self.distributor = HashDistributor(num_experts, weight_dim, seed=seed)
        self.expert_phases = np.linspace(
            0, 2 * np.pi, num_experts, endpoint=False
        ).astype(np.float32)
        self.expert_omegas = np.linspace(0.4, 3.2, num_experts).astype(np.float32)
        self.coupling_K = kuramoto_K

        rng = np.random.RandomState(seed)
        self.embedding = rng.randn(vocab_size, weight_dim).astype(np.float32) * 0.08
        self.expert_heads = None
        self.expert_proj = None
        self.built = False

    def build(
        self,
        token_ids: np.ndarray,
        vocab_size: int,
        passes: int = 2,
        primary_mass: float = 0.92,
    ):
        print(f"[PSN] building {len(token_ids):,} tokens → {self.num_experts} experts")
        result = self.distributor.distribute(
            token_ids, vocab_size, passes=passes, primary_mass=primary_mass
        )

        self.expert_heads = np.zeros((self.num_experts, vocab_size), dtype=np.float32)
        self.expert_proj = np.zeros(
            (self.num_experts, self.weight_dim, vocab_size), dtype=np.float32
        )

        for e in range(self.num_experts):
            W = result["expert_weights"][e]
            try:
                U, S, _ = np.linalg.svd(W, full_matrices=False)
                k = min(12, U.shape[1])
                proj = U[:, :k].T @ self.embedding.T
                head = (S[:k, None] * proj).sum(axis=0)
                head = head - head.min()
                if head.max() > 1e-8:
                    head /= head.max()
                self.expert_heads[e] = head.astype(np.float32)
                self.expert_proj[e] = (
                    U[:, :k] @ (np.diag(S[:k]) @ U[:, :k].T @ self.embedding.T)
                ).astype(np.float32)
            except Exception:
                pass

        self._build_ngrams(token_ids, vocab_size)
        self.built = True
        print("[PSN] ready")

    def _build_ngrams(self, token_ids: np.ndarray, vocab_size: int):
        a = token_ids[:-1].astype(np.int64)
        b = token_ids[1:].astype(np.int64)
        flat = a * vocab_size + b
        counts = np.bincount(flat, minlength=vocab_size * vocab_size)
        self.bigram = counts.reshape(vocab_size, vocab_size).astype(np.float32)
        row = self.bigram.sum(axis=1, keepdims=True)
        self.bigram_probs = self.bigram / (row + 1e-10)
        self.bigram_probs = 0.93 * self.bigram_probs + 0.07 / vocab_size

        self.trigram: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(len(token_ids) - 2):
            key = (int(token_ids[i]), int(token_ids[i + 1]))
            nxt = int(token_ids[i + 2])
            if key not in self.trigram:
                self.trigram[key] = np.zeros(vocab_size, dtype=np.float32)
            self.trigram[key][nxt] += 1.0
        for key in list(self.trigram):
            s = self.trigram[key].sum()
            if s > 0:
                self.trigram[key] = 0.88 * (self.trigram[key] / s) + 0.12 / vocab_size

        print(
            f"[PSN] n-grams: {self.bigram.sum():.0f} bigrams, {len(self.trigram)} trigrams"
        )

    def generate(
        self,
        seed_text: str,
        encode_fn,
        decode_fn,
        length: int = 250,
        temperature: float = 0.48,
        expert_strength: float = 0.78,
        use_kuramoto: bool = True,
    ) -> str:
        if not self.built:
            raise RuntimeError("call build() first")

        tokens = encode_fn(seed_text)
        cw = 4

        for _ in range(length):
            cur = tokens[-1]
            ctx = (
                tokens[-cw:]
                if len(tokens) >= cw
                else [0] * (cw - len(tokens)) + tokens
            )

            if len(tokens) >= 2:
                key = (tokens[-2], cur)
                if key in self.trigram:
                    scores = 0.62 * self.trigram[key] + 0.38 * self.bigram_probs[cur]
                else:
                    scores = self.bigram_probs[cur].copy()
            else:
                scores = self.bigram_probs[min(cur, self.vocab_size - 1)].copy()

            e = self.distributor.get_primary(cur)
            freq = self.distributor.token_freq[e]
            freq = (
                freq / (freq.max() + 1e-8)
                if freq.max() > 0
                else np.ones(self.vocab_size, dtype=np.float32) * 0.01
            )
            head = (
                self.expert_heads[e]
                if self.expert_heads is not None
                else np.zeros(self.vocab_size)
            )

            ctx_emb = self.embedding[cur].copy()
            for c in ctx[-3:]:
                ctx_emb += self.embedding[min(c, self.vocab_size - 1)]
            ctx_emb /= 4.0
            if self.expert_proj is not None:
                proj = ctx_emb @ self.expert_proj[e]
                proj = proj - proj.min()
                if proj.max() > 1e-8:
                    proj /= proj.max()
            else:
                proj = np.zeros(self.vocab_size)

            expert_signal = 0.35 * freq + 0.20 * head + 0.45 * proj

            top = np.argsort(-scores)[:22]
            for i in top:
                scores[i] *= 1.0 + expert_strength * expert_signal[i]

            for recent in tokens[-6:]:
                if 0 <= recent < len(scores):
                    scores[recent] *= 0.72

            if use_kuramoto:
                phase = self.expert_phases[e]
                for i in top[:10]:
                    tp = 2 * np.pi * i / max(self.vocab_size, 1)
                    scores[i] *= 1.0 + 0.07 * np.cos(phase - tp)

            scores = np.maximum(scores, 1e-12)
            logp = np.log(scores) / temperature
            logp -= logp.max()
            p = np.exp(logp)
            p /= p.sum()
            tokens.append(int(np.random.choice(self.vocab_size, p=p)))

            if use_kuramoto:
                self._tick()

        return decode_fn(tokens)

    def _tick(self, dt: float = 0.08):
        N = self.num_experts
        diff = self.expert_phases[None, :] - self.expert_phases[:, None]
        coup = (self.coupling_K / N) * np.sum(np.sin(diff), axis=1)
        self.expert_phases = (
            self.expert_phases + dt * (self.expert_omegas + coup)
        ) % (2 * np.pi)

    def phase_info(self) -> Dict:
        z = np.mean(np.exp(1j * self.expert_phases))
        return {
            "order_parameter": float(abs(z)),
            "phase_spread": float(
                self.expert_phases.max() - self.expert_phases.min()
            ),
        }

    def specialization_report(self, idx_to_char: Dict[int, str], top_k: int = 5):
        print("\n=== Expert specialization ===")
        for e in range(min(12, self.num_experts)):
            freq = self.distributor.token_freq[e]
            top = np.argsort(-freq)[:top_k]
            parts = []
            for i in top:
                if freq[i] <= 0:
                    continue
                ch = idx_to_char.get(int(i), "?")
                if ch == "\n":
                    ch = "\\n"
                elif ch == " ":
                    ch = "␣"
                parts.append(f"{ch}:{int(freq[i])}")
            print(
                f"E{e:2d} ({self.distributor.expert_token_count[e]:3d} toks)  "
                + ", ".join(parts)
            )
