"""Phase Siren Network (PSN) - the complete architecture.

This module assembles all components into a single coherent system:

  Token Embedding -> SIREN Phase Encoder -> Phase Hash Router
       -> Expert Ensemble (Kuramoto-coupled) -> Output Projection
       -> Kuramoto Tick -> Self-Modification Check

The network processes tokens in a continuous "ticking" fashion:
1. Tokens are embedded and phase-encoded by a SIREN stack
2. Phase fingerprints are computed for routing
3. Top-k experts are selected by phase resonance
4. Experts process tokens (phase-modulated SIREN computation)
5. Kuramoto attractor advances all expert phases by RK4
6. Self-modification rules may clone/prune/adapt experts
7. Inter-instance synchronization occurs periodically

No gradient flows through any of this. Learning happens through
phase alignment, Hebbian coupling adaptation, and structural
self-modification.

This is the complete forward pass of a PSN - it is not a loss
function and optimizer, but a dynamical system that self-organizes.
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Tuple, Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psn.core.phase_siren import PhaseSirenLayer, PhaseSirenStack
from psn.core.kuramoto import KuramotoAttractor, compute_order_parameter
from psn.core.phase_router import PhaseHashRouter
from psn.core.expert import ExpertEnsemble
from psn.config import PSNConfig


class PhaseSirenNetwork:
    """The complete Phase Siren Network.

    Assembles SIREN encoding, phase routing, expert processing, and
    Kuramoto dynamics into a single forward-pass object.

    Attributes:
        config: PSNConfig with all hyperparameters.
        token_embedding: Learnable token embedding matrix.
        siren_encoder: SIREN stack that converts embeddings to phase space.
        router: Phase hash router for expert selection.
        experts: Ensemble of expert oscillators.
        kuramoto: Kuramoto attractor managing phase dynamics.
        output_proj: Linear projection from expert output to vocab logits.
    """

    def __init__(self, config: PSNConfig, key: jax.Array = None):
        self.config = config
        if key is None:
            key = random.PRNGKey(0)

        # Split keys for all components
        keys = random.split(key, 7)

        # Token embedding: (vocab_size, embedding_dim)
        self.token_embedding = random.normal(
            keys[0], (config.vocab_size, config.embedding_dim)
        ) * 0.02

        # SIREN encoder: embedding_dim -> phase_dim
        self.siren_encoder = PhaseSirenStack(
            input_dim=config.embedding_dim,
            hidden_dim=config.siren_hidden_dim,
            output_dim=config.phase_dim,
            num_layers=3,
            omega_0=config.omega_0,
            key=keys[1],
        )

        # Phase hash router
        self.router = PhaseHashRouter(
            embedding_dim=config.phase_dim,
            hash_dim=config.hash_dim,
            num_experts=config.num_experts,
            top_k=config.routing_top_k,
            temperature=config.hash_temperature,
            key=keys[2],
        )

        # Expert ensemble
        self.experts = ExpertEnsemble(
            num_experts=config.num_experts,
            input_dim=config.phase_dim,
            output_dim=config.phase_dim,
            omega_0=config.omega_0,
            key=keys[3],
        )

        # Kuramoto attractor
        self.kuramoto = KuramotoAttractor(
            num_oscillators=config.num_experts,
            K=config.kuramoto_K,
            dt=config.dt,
            key=keys[4],
        )

        # Output projection: phase_dim -> vocab_size (for next-token prediction)
        self.output_proj_W = random.normal(
            keys[5], (config.phase_dim, config.vocab_size)
        ) * (1.0 / jnp.sqrt(config.phase_dim))
        self.output_proj_b = jnp.zeros(config.vocab_size)

        # Global state
        self.tick_count = 0
        self.step_count = 0

    def embed_tokens(self, token_ids: jax.Array) -> jax.Array:
        """Look up token embeddings.

        Args:
            token_ids: Integer token IDs, shape (batch, seq_len) or (batch,).

        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim) or (batch, embedding_dim).
        """
        return self.token_embedding[token_ids]

    def encode_to_phase(self, x: jax.Array) -> jax.Array:
        """Encode embeddings through SIREN stack to get phase representations.

        Args:
            x: Embeddings of shape (batch, embedding_dim).

        Returns:
            Phase representations of shape (batch, phase_dim).
        """
        result = self.siren_encoder(x)
        return result["phases"][-1]

    def forward_pass(
        self, token_ids: jax.Array, target_ids: Optional[jax.Array] = None
    ) -> Dict[str, Any]:
        """Complete forward pass for a batch of token sequences.

        Processes each token through: embed -> SIREN encode -> route ->
        expert process -> Kuramoto tick -> output project.

        Args:
            token_ids: Input token IDs, shape (batch, seq_len).
            target_ids: Optional target token IDs for training, shape (batch, seq_len).

        Returns:
            Dictionary with logits, routing info, Kuramoto diagnostics, etc.
        """
        batch_size, seq_len = token_ids.shape
        config = self.config

        all_logits = []
        all_routing_weights = []
        all_token_phases = []
        order_params = []
        correct_per_expert = jnp.zeros(config.num_experts)
        total_routed = jnp.zeros(config.num_experts)

        for t in range(seq_len):
            # 1. Embed current token
            token_t = token_ids[:, t]  # (batch,)
            embeddings = self.embed_tokens(token_t)  # (batch, embedding_dim)

            # 2. Encode to phase space
            phase_repr = self.encode_to_phase(embeddings)  # (batch, phase_dim)

            # 3. Get current expert phases for routing
            expert_phases = self.experts.get_phases()  # (num_active,)

            # 4. Route tokens to experts
            routing_weights, top_k_indices, token_phases = self.router.route(
                phase_repr, expert_phases
            )  # routing: (batch, num_active), top_k: (batch, top_k)

            # 5. Expert processing
            expert_output = self.experts.process_with_routing(
                phase_repr, routing_weights, top_k_indices
            )  # (batch, phase_dim)

            # 6. Output projection -> logits
            logits = jnp.einsum(
                "bi,iv->bv", expert_output, self.output_proj_W
            ) + self.output_proj_b  # (batch, vocab_size)

            # 7. Kuramoto tick (advance phase dynamics)
            omegas = self.experts.get_omegas()  # (num_active,)

            # Compute external kick from token phases
            # Each token's phase fingerprint nudges the experts that processed it
            kick = jnp.zeros_like(expert_phases)
            for k in range(config.routing_top_k):
                idx = top_k_indices[:, k]  # (batch,)
                for b in range(batch_size):
                    local_idx = int(idx[b])
                    if local_idx < expert_phases.shape[0]:
                        kick = kick.at[local_idx].add(
                            jnp.sin(token_phases[b] - expert_phases[local_idx]) * 0.01
                        )

            new_phases, diagnostics = self.kuramoto.tick(
                expert_phases, omegas,
                num_steps=config.num_ticks,
                external_kick=kick if jnp.any(jnp.abs(kick) > 1e-8) else None,
            )
            self.experts.set_phases(new_phases)

            # 8. Track metrics
            all_logits.append(logits)
            all_routing_weights.append(routing_weights)
            all_token_phases.append(token_phases)
            order_params.append(diagnostics["order_parameter"])

            # 9. Update expert accuracy if targets provided
            if target_ids is not None:
                target_t = target_ids[:, t]  # (batch,)
                predicted = jnp.argmax(logits, axis=-1)  # (batch,)
                correct_mask = (predicted == target_t).astype(float)  # (batch,)
                # Distribute credit to routed experts
                for k in range(config.routing_top_k):
                    idx = top_k_indices[:, k]
                    for b in range(batch_size):
                        local_idx = int(idx[b])
                        if local_idx < correct_per_expert.shape[0]:
                            correct_per_expert = correct_per_expert.at[local_idx].add(
                                float(correct_mask[b]) * float(routing_weights[b, local_idx])
                            )
                            total_routed = total_routed.at[local_idx].add(
                                float(routing_weights[b, local_idx])
                            )

            self.tick_count += 1

        self.step_count += 1

        # Stack logits: (seq_len, batch, vocab_size) -> (batch, seq_len, vocab_size)
        logits_seq = jnp.stack(all_logits, axis=1)

        return {
            "logits": logits_seq,
            "routing_weights": all_routing_weights,
            "token_phases": all_token_phases,
            "order_parameter_history": order_params,
            "correct_per_expert": correct_per_expert,
            "total_routed": total_routed,
            "avg_order_param": float(jnp.mean(jnp.array(order_params))),
        }

    def generate(
        self, seed_tokens: List[int], length: int, temperature: float = 0.8
    ) -> List[int]:
        """Generate text autoregressively.

        The generation process leverages the continuous phase dynamics:
        after each token, the Kuramoto system ticks forward, changing
        the expert phases and thus the routing and computation for
        the next token. This means the same seed can produce different
        outputs at different points in the network's dynamical evolution.

        Args:
            seed_tokens: List of token IDs to seed generation.
            length: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).

        Returns:
            List of generated token IDs (including seed).
        """
        generated = list(seed_tokens)
        config = self.config

        for _ in range(length):
            # Use last config.seq_len tokens as context
            context = generated[-config.seq_len:]
            # Pad if needed
            while len(context) < config.seq_len:
                context = [0] + context
            token_ids = jnp.array([context])  # (1, seq_len)

            # Forward pass
            result = self.forward_pass(token_ids)
            logits = result["logits"][0, -1, :]  # (vocab_size,) last position

            # Temperature scaling
            logits = logits / temperature

            # Sample from distribution
            probs = jax.nn.softmax(logits)
            next_token = int(jax.random.categorical(random.PRNGKey(self.tick_count), logits))
            generated.append(next_token)

        return generated

    def get_state(self) -> Dict[str, Any]:
        """Get the full state of the network for checkpointing or analysis.

        Returns:
            Dictionary with all mutable state.
        """
        return {
            "tick_count": self.tick_count,
            "step_count": self.step_count,
            "expert_phases": self.experts.get_phases(),
            "expert_omegas": self.experts.get_omegas(),
            "coupling_matrix": self.kuramoto.coupling_matrix,
            "expert_stats": self.experts.get_expert_stats(),
        }
