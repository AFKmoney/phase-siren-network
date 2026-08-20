"""Hash Distributor — direct token-to-weight injection via hashing.

Core idea: instead of training weights, CONSTRUCT them from data.

  H(token, context) → (expert_id, row, col, Δ)

For each token in the dataset:
  1. Hash the token + context to get (expert_id, position_in_weight_matrix)
  2. Directly accumulate into the expert's weights
  3. The expert now 'knows' about this token pattern

After processing the whole dataset, each expert's weights are a
hash-constructed summary of the tokens it 'owns'. No backprop,
no phase adaptation — pure data-to-weight injection.

This is like a random feature map: the hash function creates a
deterministic mapping from input space to weight space.
"""
import numpy as np
from typing import Tuple, Dict, List, Optional


class HashDistributor:
    """Distributes tokens directly into expert weights via hashing.

    The hash function H maps each (token, context) pair to:
    - Which expert should store this information
    - Where in the expert's weight matrix to put it
    - How much to add (based on frequency and context)

    This replaces training entirely: weights are constructed, not learned.
    """

    def __init__(
        self,
        num_experts: int,
        weight_dim: int,
        hash_dim: int = 64,
        num_hashes: int = 3,
        seed: int = 42,
    ):
        self.num_experts = num_experts
        self.weight_dim = weight_dim
        self.hash_dim = hash_dim
        self.num_hashes = num_hashes
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.projections = [
            rng.randn(hash_dim, weight_dim).astype(np.float32) * 0.1
            for _ in range(num_hashes)
        ]
        self.biases = [
            rng.randn(hash_dim).astype(np.float32)
            for _ in range(num_hashes)
        ]

        self.expert_weights = np.zeros(
            (num_experts, weight_dim, weight_dim), dtype=np.float32
        )
        self.expert_biases = np.zeros(
            (num_experts, weight_dim), dtype=np.float32
        )
        self.token_freq = None  # (num_experts, vocab_size)
        self.token_to_expert = {}

        # Precomputed primes for fast deterministic hashing
        self._p1 = 2654435761  # Knuth multiplicative hash
        self._p2 = 2246822519
        self._p3 = 3266489917

    def _fast_hash(self, token_id: int, ctx0: int = 0, ctx1: int = 0, ctx2: int = 0, ctx3: int = 0, seed: int = 42) -> int:
        """Fast deterministic 64-bit-ish hash without Python hash() or strings."""
        h = (token_id * self._p1) ^ (seed * self._p2)
        h = (h + ctx0 * self._p3) & 0xFFFFFFFF
        h = (h ^ (ctx1 * self._p1)) & 0xFFFFFFFF
        h = (h + ctx2 * self._p2) & 0xFFFFFFFF
        h = (h ^ (ctx3 * self._p3)) & 0xFFFFFFFF
        return h

    def _hash_token_to_expert(self, token_id: int, context_ids: List[int]) -> Tuple[int, int, int, float]:
        """Hash a token+context to (expert_id, row, col, value)."""
        ctx = (list(context_ids) + [0, 0, 0, 0])[:4]
        h = self._fast_hash(token_id, ctx[0], ctx[1], ctx[2], ctx[3], self.seed)
        expert_id = h % self.num_experts
        row = (h // self.num_experts) % self.weight_dim
        col = ((h // self.num_experts) // self.weight_dim) % self.weight_dim
        delta = 0.01
        return expert_id, row, col, delta

    def distribute_dataset(
        self,
        token_ids: np.ndarray,
        vocab_size: int,
        context_window: int = 4,
        passes: int = 3,
    ) -> Dict[str, np.ndarray]:
        """Process entire dataset: hash each token into expert weights.

        Vectorized core path using np.add.at for accumulation.
        """
        N = len(token_ids)
        self.token_freq = np.zeros((self.num_experts, vocab_size), dtype=np.float32)
        token_ids = token_ids.astype(np.int64)

        print(f"[HashDistributor] Processing {N} tokens x {passes} passes (vectorized)...")

        for pass_idx in range(passes):
            seed = 42 + pass_idx * 1000
            self.seed = seed

            # Pre-allocate arrays for this pass
            # We process in chunks to keep memory reasonable
            chunk_size = 200_000
            for start in range(context_window, N, chunk_size):
                end = min(start + chunk_size, N)
                idx = np.arange(start, end)

                toks = token_ids[idx]
                # Context: last 4 tokens (pad with 0)
                c0 = token_ids[np.maximum(idx - 1, 0)]
                c1 = token_ids[np.maximum(idx - 2, 0)]
                c2 = token_ids[np.maximum(idx - 3, 0)]
                c3 = token_ids[np.maximum(idx - 4, 0)]

                # Vectorized hash
                h = (toks * self._p1) ^ (seed * self._p2)
                h = (h + c0 * self._p3) & 0xFFFFFFFF
                h = (h ^ (c1 * self._p1)) & 0xFFFFFFFF
                h = (h + c2 * self._p2) & 0xFFFFFFFF
                h = (h ^ (c3 * self._p3)) & 0xFFFFFFFF

                expert_ids = (h % self.num_experts).astype(np.int32)
                rows = ((h // self.num_experts) % self.weight_dim).astype(np.int32)
                cols = (((h // self.num_experts) // self.weight_dim) % self.weight_dim).astype(np.int32)

                # Accumulate into expert weights with np.add.at
                np.add.at(self.expert_weights, (expert_ids, rows, cols), 0.01)
                np.add.at(self.expert_biases, (expert_ids, rows), 0.001)

                # Token frequency per expert
                np.add.at(self.token_freq, (expert_ids, toks), 1.0)

                # First-seen mapping (only on first pass for stability)
                if pass_idx == 0:
                    for t, e in zip(toks, expert_ids):
                        if int(t) not in self.token_to_expert:
                            self.token_to_expert[int(t)] = int(e)

            print(f"  Pass {pass_idx + 1}/{passes} done")

        # Normalize by sqrt frequency
        for e in range(self.num_experts):
            freq_sum = self.token_freq[e].sum()
            if freq_sum > 0:
                self.expert_weights[e] /= (freq_sum ** 0.5)
                self.expert_biases[e] /= (freq_sum ** 0.25)

        print(f"[HashDistributor] Done. Weight stats:")
        for e in range(min(5, self.num_experts)):
            w_norm = np.linalg.norm(self.expert_weights[e])
            tokens_owned = int((self.token_freq[e] > 0).sum())
            print(f"  Expert {e}: ||W||={w_norm:.4f}, tokens={tokens_owned}")

        return {
            "expert_weights": self.expert_weights,
            "expert_biases": self.expert_biases,
            "token_freq": self.token_freq,
            "token_to_expert": self.token_to_expert,
        }

    def get_expert_for_token(self, token_id: int, context_ids: List[int] = None) -> int:
        if token_id in self.token_to_expert:
            return self.token_to_expert[token_id]
        if context_ids is None:
            context_ids = [0]
        expert_id, _, _, _ = self._hash_token_to_expert(token_id, context_ids)
        return expert_id

    def get_next_token_scores(
        self,
        current_token_id: int,
        context_ids: List[int],
        vocab_size: int,
    ) -> np.ndarray:
        """Score next tokens from the hash-distributed expert frequency tables."""
        current_expert = self.get_expert_for_token(current_token_id, context_ids)
        scores = self.token_freq[current_expert].copy()

        # Soft routing to neighbors
        for delta in (-1, 1):
            neighbor = (current_expert + delta) % self.num_experts
            scores += self.token_freq[neighbor] * 0.25

        scores = np.log1p(scores)
        return scores


class HashBasedPSN:
    """Complete PSN with hash-distributed weights + Kuramoto dynamics.

    Weights are CONSTRUCTED by hashing, not learned by gradient.
    Kuramoto provides temporal dynamics and soft modulation on top.
    """

    def __init__(
        self,
        num_experts: int = 32,
        weight_dim: int = 64,
        hash_dim: int = 64,
        kuramoto_K: float = 2.0,
        vocab_size: int = 65,
        seed: int = 42,
    ):
        self.num_experts = num_experts
        self.weight_dim = weight_dim
        self.vocab_size = vocab_size
        self.seed = seed

        self.distributor = HashDistributor(
            num_experts=num_experts,
            weight_dim=weight_dim,
            hash_dim=hash_dim,
            seed=seed,
        )

        self.expert_phases = np.linspace(0, 2 * np.pi, num_experts, endpoint=False).astype(np.float32)
        self.expert_omegas = np.linspace(0.5, 3.0, num_experts).astype(np.float32)
        self.coupling_K = kuramoto_K

        rng = np.random.RandomState(seed)
        self.embedding = rng.randn(vocab_size, weight_dim).astype(np.float32) * 0.1
        self.output_W = rng.randn(weight_dim, vocab_size).astype(np.float32) * 0.05

        # Per-expert output heads derived from the constructed weights
        self.expert_output_heads = None  # (num_experts, weight_dim, vocab_size) after build

        self.built = False

    def build_from_data(self, token_ids: np.ndarray, vocab_size: int, passes: int = 3):
        """Build the model by hashing the entire dataset into weights."""
        print(f"[HashPSN] Building from {len(token_ids)} tokens...")

        result = self.distributor.distribute_dataset(
            token_ids, vocab_size, context_window=4, passes=passes
        )

        # Build per-expert soft preference vectors from the constructed weight matrices
        # via top singular vectors. These act as a weak prior; n-gram + token_freq
        # remain the dominant signal so generation stays coherent.
        self.expert_output_heads = np.zeros(
            (self.num_experts, vocab_size), dtype=np.float32
        )
        for e in range(self.num_experts):
            W = result["expert_weights"][e]
            try:
                U, S, _ = np.linalg.svd(W, full_matrices=False)
                k = min(8, U.shape[1])
                proj = U[:, :k].T @ self.embedding.T          # (k, vocab)
                head = (S[:k, None] * proj).sum(axis=0)
                head = head - head.min()
                if head.max() > 0:
                    head = head / head.max()
                self.expert_output_heads[e] = head.astype(np.float32)
            except Exception:
                self.expert_output_heads[e] = np.zeros(vocab_size, dtype=np.float32)

        # Fast bigram / trigram construction (vectorized where possible)
        self._build_bigram_table(token_ids, vocab_size)

        self.built = True
        print(f"[HashPSN] Build complete.")

    def _build_bigram_table(self, token_ids: np.ndarray, vocab_size: int):
        """Build bigram + trigram tables (still useful as a strong base signal)."""
        # Vectorized bigram via bincount-style accumulation
        self.bigram = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        a = token_ids[:-1].astype(np.int64)
        b = token_ids[1:].astype(np.int64)
        # Use a flat index for speed
        flat = a * vocab_size + b
        counts = np.bincount(flat, minlength=vocab_size * vocab_size)
        self.bigram = counts.reshape(vocab_size, vocab_size).astype(np.float32)

        row_sums = self.bigram.sum(axis=1, keepdims=True)
        self.bigram_probs = self.bigram / (row_sums + 1e-10)
        self.bigram_probs = 0.95 * self.bigram_probs + 0.05 / vocab_size

        # Trigram remains dict-based (sparse)
        self.trigram = {}
        for i in range(len(token_ids) - 2):
            key = (int(token_ids[i]), int(token_ids[i + 1]))
            nxt = int(token_ids[i + 2])
            if key not in self.trigram:
                self.trigram[key] = np.zeros(vocab_size, dtype=np.float32)
            self.trigram[key][nxt] += 1.0
        for key in self.trigram:
            s = self.trigram[key].sum()
            if s > 0:
                self.trigram[key] /= s
                self.trigram[key] = 0.9 * self.trigram[key] + 0.1 / vocab_size

        print(f"[HashPSN] Bigram: {self.bigram.sum():.0f} pairs, Trigram: {len(self.trigram)} contexts")

    def generate(
        self,
        seed_text: str,
        encode_fn,
        decode_fn,
        length: int = 200,
        temperature: float = 0.8,
        use_kuramoto: bool = True,
        expert_weight: float = 0.18,
    ) -> str:
        """Generate text using hash-distributed expert weights + n-gram + Kuramoto.

        expert_weight controls how strongly the constructed expert heads influence
        the next-token scores (vs pure bigram/trigram).
        """
        if not self.built:
            raise RuntimeError("Must call build_from_data() first")

        tokens = encode_fn(seed_text)
        context_window = 4

        for _ in range(length):
            current_token = tokens[-1]
            context = tokens[-context_window:] if len(tokens) >= context_window else [0] * (context_window - len(tokens)) + tokens

            # Base signal from n-grams (strong local statistics)
            if len(tokens) >= 2:
                tg_key = (tokens[-2], current_token)
                if tg_key in self.trigram:
                    scores = 0.55 * self.trigram[tg_key] + 0.45 * self.bigram_probs[current_token]
                else:
                    scores = self.bigram_probs[current_token].copy()
            elif current_token < self.vocab_size:
                scores = self.bigram_probs[current_token].copy()
            else:
                scores = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

            # === Expert contribution (hash-constructed knowledge) ===
            # Applied as a multiplicative bias on the strongest n-gram candidates
            # so the sharp local statistics are preserved while still using the
            # expert structure built by the hash.
            expert_id = self.distributor.get_expert_for_token(current_token, context)
            expert_freq = self.distributor.token_freq[expert_id]
            if expert_freq.max() > 0:
                expert_freq = expert_freq / expert_freq.max()
            else:
                expert_freq = np.ones(self.vocab_size, dtype=np.float32)

            if self.expert_output_heads is not None:
                head = self.expert_output_heads[expert_id]
            else:
                head = np.zeros(self.vocab_size, dtype=np.float32)

            # Soft expert prior
            expert_prior = 0.7 * expert_freq + 0.3 * head

            # Multiplicative nudge only on the top candidates of the n-gram
            top_k = np.argsort(-scores)[:16]
            for idx in top_k:
                scores[idx] *= (1.0 + expert_weight * expert_prior[idx])

            scores = np.maximum(scores, 1e-10)

            # Light Kuramoto phase modulation on top-k only
            if use_kuramoto:
                phase = self.expert_phases[expert_id]
                top_k_idx = np.argsort(-scores)[:12]
                for idx in top_k_idx:
                    token_phase = 2 * np.pi * idx / self.vocab_size
                    nudge = 1.0 + 0.08 * np.cos(phase - token_phase)
                    scores[idx] *= nudge

            # Temperature + sample
            log_scores = np.log(np.maximum(scores, 1e-12)) / temperature
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            probs /= probs.sum() + 1e-12

            next_token = int(np.random.choice(self.vocab_size, p=probs))
            tokens.append(next_token)

            if use_kuramoto:
                self._kuramoto_tick()

        return decode_fn(tokens)

    def _kuramoto_tick(self, dt: float = 0.1):
        """Advance Kuramoto phases (simplified Euler for speed)."""
        N = len(self.expert_phases)
        K = self.coupling_K
        diff = self.expert_phases[None, :] - self.expert_phases[:, None]
        coupling = (K / N) * np.sum(np.sin(diff), axis=1)
        self.expert_phases = (self.expert_phases + dt * (self.expert_omegas + coupling)) % (2 * np.pi)

    def get_phase_info(self) -> Dict:
        z = np.mean(np.exp(1j * self.expert_phases))
        return {
            "order_parameter": float(abs(z)),
            "mean_phase": float(np.angle(z)),
            "phase_spread": float(self.expert_phases.max() - self.expert_phases.min()),
            "expert_phases": self.expert_phases.tolist(),
            "expert_omegas": self.expert_omegas.tolist(),
        }
