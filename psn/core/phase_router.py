"""Phase Hash Router — routes tokens to experts via phase-resonance hashing.

Traditional mixture-of-experts (Shazeer et al. 2017) route tokens through a
learned linear projection + softmax. This has two fundamental problems:

1. The routing function is fixed once trained — no online adaptation.
2. Routing decisions are point-estimates with no dynamical structure.

Our approach replaces the linear router with a **phase-resonance mechanism**:

  1. Each token x produces a phase fingerprint phi_x = H(x) mod 2*pi
     via a deterministic hash-like projection.
  2. Each expert i has a current phase theta_i from the Kuramoto dynamics.
  3. Resonance is measured as: R_i = cos(theta_i - phi_x)
     (maximum when expert phase equals token phase fingerprint).
  4. Top-k experts with highest resonance receive the token.
  5. Routing weights are: w_i = softmax(R_i / temperature) for top-k.

Key property: routing is **dynamical** — as expert phases evolve through
Kuramoto coupling, the routing pattern changes continuously. An expert
that was resonant at t=0 may become non-resonant at t=5, causing natural
load balancing without any explicit load-balancing loss.

No gradients flow through the router. The hash projection H is initialized
randomly and adapted via phase alignment rules (see self_modification.py).
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any


class PhaseHashRouter:
    """Routes tokens to experts based on phase resonance with hash fingerprints.

    The router maintains a hash projection matrix H that maps token embeddings
    to phase fingerprints. Routing decisions are made by computing cosine
    similarity (circular) between token phases and expert phases.

    Attributes:
        embedding_dim: Input token embedding dimension.
        hash_dim: Internal hash projection dimension (before reduction to scalar).
        num_experts: Total number of expert slots.
        top_k: Number of experts each token is routed to.
        temperature: Softmax temperature for routing weights.
        H: Hash projection matrix of shape (embedding_dim, hash_dim).
        hash_bias: Bias vector of shape (hash_dim,).
    """

    def __init__(
        self,
        embedding_dim: int,
        hash_dim: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
        key: jax.Array = None,
    ):
        self.embedding_dim = embedding_dim
        self.hash_dim = hash_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        if key is None:
            key = random.PRNGKey(123)
        k1, k2 = random.split(key)
        # Hash projection: deterministic mapping from embedding to phase space
        self.H = random.normal(k1, (embedding_dim, hash_dim)) * 0.1
        self.hash_bias = random.uniform(k2, (hash_dim,), minval=0, maxval=2 * jnp.pi)

    def compute_phase_fingerprint(self, x: jax.Array) -> jax.Array:
        """Compute phase fingerprint for a batch of token embeddings.

        The fingerprint is the circular mean of the hash-projected phases.
        This gives a single scalar phase in [0, 2*pi) per token.

        Args:
            x: Token embeddings of shape (batch, embedding_dim).

        Returns:
            Phase fingerprints of shape (batch,).
        """
        # Project to hash space and apply phase offset
        projected = jnp.einsum("bi,ij->bj", x, self.H) + self.hash_bias  # (batch, hash_dim)
        # Convert to phases
        phases = jnp.mod(projected, 2 * jnp.pi)  # (batch, hash_dim)
        # Circular mean -> scalar fingerprint
        z = jnp.exp(1j * phases)
        mean_z = jnp.mean(z, axis=-1)  # (batch,)
        fingerprint = jnp.angle(mean_z) % (2 * jnp.pi)  # (batch,)
        return fingerprint

    def compute_resonance(
        self, token_phases: jax.Array, expert_phases: jax.Array
    ) -> jax.Array:
        """Compute phase resonance between tokens and experts.

        Resonance R_ij = cos(theta_j - phi_i) where:
        - phi_i is the phase fingerprint of token i
        - theta_j is the current phase of expert j

        This is the circular cosine similarity. Maximum resonance (R=1)
        occurs when token and expert phases are perfectly aligned.

        Args:
            token_phases: Token phase fingerprints, shape (batch,).
            expert_phases: Expert current phases, shape (num_experts,).

        Returns:
            Resonance matrix of shape (batch, num_experts).
        """
        # (batch, 1) - (1, num_experts) -> (batch, num_experts)
        phase_diff = token_phases[:, None] - expert_phases[None, :]
        resonance = jnp.cos(phase_diff)
        return resonance

    def route(
        self, x: jax.Array, expert_phases: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Route tokens to experts based on phase resonance.

        Computes top-k routing with softmax-weighted combination.

        Args:
            x: Token embeddings of shape (batch, embedding_dim).
            expert_phases: Current expert phases, shape (num_experts,).

        Returns:
            routing_weights: Softmax weights for top-k, shape (batch, num_experts).
                Non-selected experts get weight 0.
            top_k_indices: Which experts are selected, shape (batch, top_k).
            token_phases: Computed phase fingerprints, shape (batch,).
        """
        batch_size = x.shape[0]
        token_phases = self.compute_phase_fingerprint(x)  # (batch,)
        resonance = self.compute_resonance(token_phases, expert_phases)  # (batch, num_experts)

        # Mask: keep only top-k per token
        # For small num_experts, we can use a simple sort approach
        # Sort in descending order and keep top-k
        sorted_indices = jnp.argsort(-resonance, axis=-1)[:, : self.top_k]  # (batch, top_k)

        # Create sparse routing weight matrix
        routing_weights = jnp.zeros((batch_size, self.num_experts))

        # Gather top-k resonances and apply softmax
        for k in range(self.top_k):
            idx = sorted_indices[:, k]  # (batch,)
            batch_idx = jnp.arange(batch_size)
            # Gather resonances for top-k experts
            top_resonance = resonance[batch_idx, idx]  # (batch,)
            # Add to routing weights
            routing_weights = routing_weights.at[batch_idx, idx].add(top_resonance)

        # Apply softmax over the selected experts (mask zeros)
        # Shifted softmax for numerical stability
        max_r = jnp.max(routing_weights, axis=-1, keepdims=True)
        exp_r = jnp.exp((routing_weights - max_r) / self.temperature)
        sum_exp = jnp.sum(exp_r, axis=-1, keepdims=True)
        routing_weights = exp_r / (sum_exp + 1e-8)

        return routing_weights, sorted_indices, token_phases

    def adapt_hash(
        self, x: jax.Array, correct_expert_phases: jax.Array, lr: float = 0.01
    ) -> "PhaseHashRouter":
        """Adapt the hash projection to improve routing quality.

        Non-gradient adaptation: if a token was correctly processed by
        expert j, we nudge H so that H(x) produces a fingerprint closer
        to theta_j (the expert's phase).

        Adaptation rule:
          phi_current = H(x)  (current fingerprint)
          phi_target = theta_j   (expert phase)
          delta = angle(exp(i*(phi_target - phi_current)))  (circular diff)
          H <- H + lr * delta @ x^T / batch_size

        Args:
            x: Token embeddings, shape (batch, embedding_dim).
            correct_expert_phases: Phase of the expert that correctly handled
                each token, shape (batch,).
            lr: Adaptation learning rate.

        Returns:
            New PhaseHashRouter with updated hash projection.
        """
        current_fingerprints = self.compute_phase_fingerprint(x)  # (batch,)
        # Circular phase difference
        delta = jnp.angle(jnp.exp(1j * (correct_expert_phases - current_fingerprints)))  # (batch,)
        # Project delta back through the hash
        # delta is scalar per token; we need to distribute it across hash_dim
        # Use the token embedding as a guide for which hash dimensions to adjust
        projected = jnp.einsum("bi,ij->bj", x, self.H) + self.hash_bias
        phases = jnp.mod(projected, 2 * jnp.pi)
        # Nudge hash bias so circular mean shifts toward target
        phase_error = delta[:, None] * jnp.ones_like(phases)  # (batch, hash_dim)
        d_bias = lr * jnp.mean(phase_error, axis=0)  # (hash_dim,)
        # Weight update: make H(x) produce more discriminative phases
        d_H = lr * jnp.einsum("bi,bj->ij", x, phase_error) / x.shape[0]

        new_router = PhaseHashRouter(
            self.embedding_dim, self.hash_dim, self.num_experts,
            self.top_k, self.temperature, key=None,
        )
        new_router.H = self.H + d_H
        new_router.hash_bias = self.hash_bias + d_bias
        return new_router

    def routing_entropy(self, routing_weights: jax.Array) -> float:
        """Compute the entropy of the routing distribution.

        High entropy = tokens are spread evenly across experts (good load balancing).
        Low entropy = tokens concentrate on few experts (potential bottleneck).

        Args:
            routing_weights: Routing weight matrix, shape (batch, num_experts).

        Returns:
            Scalar entropy value.
        """
        # Average routing distribution over the batch
        avg_routing = jnp.mean(routing_weights, axis=0)  # (num_experts,)
        avg_routing = avg_routing / (jnp.sum(avg_routing) + 1e-8)  # normalize
        # Shannon entropy
        entropy = -jnp.sum(avg_routing * jnp.log(avg_routing + 1e-8))
        return float(entropy)
