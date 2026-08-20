"""Hash Distributor — direct token-to-weight injection via hashing.

Core idea: instead of training weights, CONSTRUCT them from data.

  H(token, context) → (expert_id, row, col, delta)

For each token in the dataset:
  1. Hash the token + context to get (expert_id, position_in_weight_matrix)
  2. Directly accumulate the token's embedding into the expert's weights
  3. The expert now 'knows' about this token pattern

After processing the whole dataset, each expert's weights are a
hash-constructed summary of the tokens it 'owns'. No backprop,
no phase adaptation — pure data-to-weight injection.

This is like a random feature map: the hash function creates a
deterministic mapping from input space to weight space.
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
import hashlib
import struct


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
        
        # Random projection matrices for hashing (deterministic via seed)
        rng = np.random.RandomState(seed)
        # One projection per hash function
        self.projections = [
            rng.randn(hash_dim, weight_dim).astype(np.float32) * 0.1
            for _ in range(num_hashes)
        ]
        # Biases for each hash
        self.biases = [
            rng.randn(hash_dim).astype(np.float32)
            for _ in range(num_hashes)
        ]
        
        # Expert weight accumulators: (num_experts, weight_dim, weight_dim)
        # These get filled directly from data
        self.expert_weights = np.zeros(
            (num_experts, weight_dim, weight_dim), dtype=np.float32
        )
        # Expert bias accumulators: (num_experts, weight_dim)
        self.expert_biases = np.zeros(
            (num_experts, weight_dim), dtype=np.float32
        )
        # Token frequency per expert: (num_experts, vocab_size)
        self.token_freq = None  # Built during distribute()
        
        # Track which tokens map to which expert
        self.token_to_expert = {}
    
    def _hash_token_to_expert(self, token_id: int, context_ids: List[int]) -> Tuple[int, int, int, float]:
        """Hash a token+context to (expert_id, row, col, value).
        
        Uses a deterministic hash that maps the same token+context
        to the same expert and weight position every time.
        
        Args:
            token_id: Integer token ID.
            context_ids: List of preceding token IDs (context window).
            
        Returns:
            (expert_id, row, col, delta_value) tuple.
        """
        # Create a deterministic string from token + context
        ctx_str = ",".join(str(c) for c in context_ids[-4:])  # Last 4 for context
        key = f"{token_id}:{ctx_str}:{self.seed}"
        
        # Use Python's hash (deterministic within a process)
        h = hash(key)
        
        # Expert selection from hash
        expert_id = h % self.num_experts
        
        # Position in weight matrix
        row = (h // self.num_experts) % self.weight_dim
        col = ((h // self.num_experts) // self.weight_dim) % self.weight_dim
        
        # Value: small constant, will accumulate over many tokens
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
        
        This is the 'training' — but it's just hash + accumulate.
        No gradient, no iteration, no epochs.
        
        Args:
            token_ids: 1D array of all token IDs in the dataset.
            vocab_size: Vocabulary size.
            context_window: How many preceding tokens to include in hash.
            passes: Number of passes over the data (more passes = richer weights).
            
        Returns:
            Dictionary with expert_weights, expert_biases, token_freq.
        """
        N = len(token_ids)
        self.token_freq = np.zeros((self.num_experts, vocab_size), dtype=np.float32)
        
        print(f"[HashDistributor] Processing {N} tokens x {passes} passes...")
        
        for pass_idx in range(passes):
            # Vary the seed slightly per pass for richer coverage
            self.seed = 42 + pass_idx * 1000
            
            for i in range(context_window, N):
                token_id = int(token_ids[i])
                context = token_ids[max(0, i - context_window):i].tolist()
                
                # Hash to get expert + position
                expert_id, row, col, delta = self._hash_token_to_expert(
                    token_id, context
                )
                
                # Also hash context tokens to nearby positions
                for ctx_offset, ctx_id in enumerate(context):
                    _, ctx_row, ctx_col, ctx_delta = self._hash_token_to_expert(
                        int(ctx_id), context[:ctx_offset] if ctx_offset > 0 else [0]
                    )
                    # Add context information to same expert
                    self.expert_weights[expert_id, ctx_row, ctx_col] += ctx_delta * 0.5
                
                # Accumulate into expert weights (direct injection)
                self.expert_weights[expert_id, row, col] += delta
                self.expert_biases[expert_id, row] += delta * 0.1
                
                # Track token frequency per expert
                self.token_freq[expert_id, token_id] += 1.0
                
                # Store mapping
                if token_id not in self.token_to_expert:
                    self.token_to_expert[token_id] = expert_id
            
            print(f"  Pass {pass_idx + 1}/{passes} done")
        
        # Normalize weights by frequency to prevent explosion
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
        """Get which expert handles a given token.
        
        Args:
            token_id: Token to look up.
            context_ids: Optional context for context-aware routing.
            
        Returns:
            Expert ID.
        """
        if token_id in self.token_to_expert:
            return self.token_to_expert[token_id]
        # Fallback: hash-based
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
        """Score each possible next token based on hash-distributed weights.
        
        For each candidate next token, check if it 'belongs' to the same
        expert pattern as the current context. Higher co-occurrence = higher score.
        
        Args:
            current_token_id: Current token.
            context_ids: Preceding tokens.
            vocab_size: Vocabulary size.
            
        Returns:
            Score array of shape (vocab_size,).
        """
        # Find which expert handles the current context
        current_expert = self.get_expert_for_token(current_token_id, context_ids)
        
        # Score: how often each token appears in this expert's frequency table
        scores = self.token_freq[current_exp].copy()
        
        # Also consider neighboring experts for soft routing
        for delta in [-1, 1]:
            neighbor = (current_expert + delta) % self.num_experts
            scores += self.token_freq[neighbor] * 0.3
        
        # Temperature scaling (softer distribution)
        scores = np.log1p(scores)  # log(1+x) for smoothing
        
        return scores


class HashBasedPSN:
    """Complete PSN with hash-distributed weights + Kuramoto dynamics.
    
    This combines:
    1. HashDistributor: constructs expert weights directly from data
    2. Kuramoto attractor: provides temporal dynamics for generation
    3. Phase routing: selects experts based on phase resonance
    
    The key difference from the original PSN: weights are CONSTRUCTED
    by hashing, not LEARNED by adaptation. The Kuramoto dynamics
    add temporal structure on top.
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
        
        # Hash distributor: builds weights from data
        self.distributor = HashDistributor(
            num_experts=num_experts,
            weight_dim=weight_dim,
            hash_dim=hash_dim,
            seed=seed,
        )
        
        # Kuramoto state for each expert
        self.expert_phases = np.linspace(0, 2 * np.pi, num_experts, endpoint=False).astype(np.float32)
        self.expert_omegas = np.linspace(0.5, 3.0, num_experts).astype(np.float32)
        self.coupling_K = kuramoto_K
        
        # Token-to-embedding (simple one-hot + projection)
        rng = np.random.RandomState(seed)
        self.embedding = rng.randn(vocab_size, weight_dim).astype(np.float32) * 0.1
        
        # Output mapping: expert weights -> vocab scores
        self.output_W = rng.randn(weight_dim, vocab_size).astype(np.float32) * 0.05
        
        self.built = False
    
    def build_from_data(self, token_ids: np.ndarray, vocab_size: int, passes: int = 3):
        """Build the model by hashing the entire dataset into weights.
        
        This replaces training. One call, and the model is 'trained'.
        
        Args:
            token_ids: 1D array of all token IDs.
            vocab_size: Vocabulary size.
            passes: Number of hash passes for richer weight construction.
        """
        print(f"[HashPSN] Building from {len(token_ids)} tokens...")
        
        # Step 1: Hash-distribute tokens into expert weights
        result = self.distributor.distribute_dataset(
            token_ids, vocab_size, context_window=4, passes=passes
        )
        
        # Step 2: Build output projection from expert weights
        # Each expert's weight matrix defines a subspace.
        # The output projection maps from this subspace to vocab.
        for e in range(self.num_experts):
            # Use SVD to get the principal direction of each expert
            W = result["expert_weights"][e]
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            # The top singular vectors define the expert's output mapping
            expert_output = U[:, :min(10, U.shape[1])]  # Top 10 components
            # Accumulate into global output projection
            self.output_W += expert_output @ np.random.randn(min(10, U.shape[1]), vocab_size).astype(np.float32) * 0.01
        
        # Step 3: Also build a bigram table from the data for direct next-token scoring
        self._build_bigram_table(token_ids, vocab_size)
        
        self.built = True
        print(f"[HashPSN] Build complete.")
    
    def _build_bigram_table(self, token_ids: np.ndarray, vocab_size: int):
        """Build bigram + trigram tables from data.
        
        This IS the hash function at work: co-occurrence counting is
        the simplest deterministic hash. H(a, b) -> count(a, b).
        """
        # Bigram
        self.bigram = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        for i in range(len(token_ids) - 1):
            self.bigram[int(token_ids[i]), int(token_ids[i + 1])] += 1.0
        row_sums = self.bigram.sum(axis=1, keepdims=True)
        self.bigram_probs = self.bigram / (row_sums + 1e-10)
        # Light smoothing
        self.bigram_probs = 0.95 * self.bigram_probs + 0.05 / vocab_size
        
        # Trigram: P(c | a, b) stored as (a, b, c)
        self.trigram = {}
        for i in range(len(token_ids) - 2):
            key = (int(token_ids[i]), int(token_ids[i + 1]))
            nxt = int(token_ids[i + 2])
            if key not in self.trigram:
                self.trigram[key] = np.zeros(vocab_size, dtype=np.float32)
            self.trigram[key][nxt] += 1.0
        # Normalize trigram entries
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
    ) -> str:
        """Generate text using hash-distributed weights + Kuramoto dynamics.
        
        Generation process:
        1. Hash each context token to find the active expert
        2. Use bigram table for base next-token scores
        3. Modulate scores by expert phase (Kuramoto dynamics)
        4. Sample next token
        5. Tick Kuramoto system
        
        Args:
            seed_text: Seed string to start generation.
            encode_fn: Function str -> List[int].
            decode_fn: Function List[int] -> str.
            length: Characters to generate.
            temperature: Sampling temperature.
            use_kuramoto: Whether to use Kuramoto phase dynamics.
            
        Returns:
            Generated text string.
        """
        if not self.built:
            raise RuntimeError("Must call build_from_data() first")
        
        tokens = encode_fn(seed_text)
        context_window = 4
        
        for _ in range(length):
            current_token = tokens[-1]
            context = tokens[-context_window:] if len(tokens) >= context_window else [0] * (context_window - len(tokens)) + tokens
            
            # Try trigram first (more specific), fall back to bigram
            if len(tokens) >= 2:
                tg_key = (tokens[-2], current_token)
                if tg_key in self.trigram:
                    # Mix trigram and bigram: trigram is more specific but sparser
                    scores = 0.6 * self.trigram[tg_key] + 0.4 * self.bigram_probs[current_token]
                else:
                    scores = self.bigram_probs[current_token].copy()
            elif current_token < self.vocab_size:
                scores = self.bigram_probs[current_token].copy()
            else:
                scores = np.ones(self.vocab_size) / self.vocab_size
            
            # Light Kuramoto modulation: use phase to softly boost top-k candidates
            if use_kuramoto:
                expert_id = self.distributor.get_expert_for_token(current_token, context)
                phase = self.expert_phases[expert_id]
                # Only modulate the TOP candidates (don't destroy the bigram signal)
                top_k_idx = np.argsort(-scores)[:8]
                for idx in top_k_idx:
                    # Each candidate gets a small phase-dependent nudge
                    token_phase = 2 * np.pi * idx / self.vocab_size
                    nudge = 1.0 + 0.05 * np.cos(phase - token_phase)
                    scores[idx] *= nudge
            
            # Temperature
            log_scores = np.log(scores + 1e-10) / temperature
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            probs /= probs.sum() + 1e-10
            
            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs)
            tokens.append(int(next_token))
            
            # Tick Kuramoto
            if use_kuramoto:
                self._kuramoto_tick()
        
        return decode_fn(tokens)
    
    def _kuramoto_tick(self, dt: float = 0.1):
        """Advance Kuramoto phases by one RK4 step.
        
        Uses simplified Kuramoto for speed during generation.
        """
        N = len(self.expert_phases)
        K = self.coupling_K
        
        # Phase differences
        diff = self.expert_phases[None, :] - self.expert_phases[:, None]
        # Coupling
        coupling = K / N * np.sum(np.sin(diff), axis=1)
        # Update
        self.expert_phases = (self.expert_phases + dt * (self.expert_omegas + coupling)) % (2 * np.pi)
    
    def get_phase_info(self) -> Dict:
        """Get current Kuramoto phase state."""
        z = np.mean(np.exp(1j * self.expert_phases))
        return {
            "order_parameter": float(abs(z)),
            "mean_phase": float(np.angle(z)),
            "phase_spread": float(self.expert_phases.max() - self.expert_phases.min()),
            "expert_phases": self.expert_phases.tolist(),
            "expert_omegas": self.expert_omegas.tolist(),
        }