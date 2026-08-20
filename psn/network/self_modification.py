"""Self-Modification System -- the network rewires itself.

This is the most speculative and novel component. The PSN can:

1. **Adapt coupling**: Hebbian-like coupling matrix updates
   (strengthen connections between co-active, phase-aligned experts)

2. **Adapt frequencies**: Nudge expert natural frequencies
   based on task performance

3. **Clone experts**: When order parameter is too high (over-synchronization),
   clone the best expert with a phase offset to restore diversity

4. **Prune experts**: Deactivate experts with very low utilization

5. **Adapt SIREN weights**: Phase-aligned weight updates
   (not gradient-based -- direct phase nudging)

6. **Adapt hash router**: Improve routing by aligning hash fingerprints
   with expert phases

7. **Adapt output projection**: Align output logits with correct targets
   using phase-based weight updates

All adaptations use the **phase alignment** principle:
  delta = angle(exp(i*(target_phase - current_phase)))
  W <- W + lr * f(delta, X)

This is fundamentally different from gradient descent:
- No loss function is differentiated
- No backpropagation chain
- Adaptations are local (expert-centric) and phase-based
- The system self-organizes through dynamical attractors
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Optional, List


class SelfModifier:
    """Orchestrates all self-modification mechanisms of the PSN.

    The modifier observes the network's state (phases, utilization,
    accuracy, order parameter) and applies adaptation rules.

    Attributes:
        adaptation_lr: Base learning rate for phase adaptations.
        freq_lr: Learning rate for natural frequency adaptation.
        clone_threshold: Order parameter above which cloning is triggered.
        prune_threshold: Utilization fraction below which pruning occurs.
        modification_log: History of all modifications for analysis.
    """

    def __init__(
        self,
        adaptation_lr: float = 0.005,
        freq_lr: float = 0.001,
        clone_threshold: float = 0.95,
        prune_threshold: float = 0.05,
    ):
        self.adaptation_lr = adaptation_lr
        self.freq_lr = freq_lr
        self.clone_threshold = clone_threshold
        self.prune_threshold = prune_threshold
        self.modification_log: List[Dict[str, Any]] = []

    def adapt_coupling(self, kuramoto, phases, expert_utils):
        """Adapt Kuramoto coupling matrix (Hebbian phase rule).

        Strengthen coupling between co-utilized, phase-aligned experts.
        Weaken coupling between phase-anti-aligned experts.

        Args:
            kuramoto: KuramotoAttractor instance.
            phases: Current expert phases.
            expert_utils: Utilization counts per expert.

        Returns:
            Updated coupling matrix.
        """
        new_coupling = kuramoto.adapt_coupling(
            phases, expert_utils, lr=self.adaptation_lr
        )
        self.modification_log.append({
            "type": "coupling_adaptation",
            "mean_coupling": float(jnp.mean(new_coupling)),
            "max_coupling": float(jnp.max(new_coupling)),
        })
        return new_coupling

    def adapt_frequencies(self, experts, phases, target_phases):
        """Adapt expert natural frequencies based on phase error.

        If an expert's phase is far from where it should be (based
        on the tokens it processed), nudge its natural frequency
        to reduce this discrepancy.

        Rule:
          delta = angle(exp(i*(target - current)))
          omega_i <- omega_i + freq_lr * delta

        Args:
            experts: ExpertEnsemble instance.
            phases: Current phases.
            target_phases: Desired phases (from task feedback).
        """
        active_experts = [e for e in experts.experts if e.active]
        N = len(active_experts)
        if N == 0:
            return

        N_target = min(N, target_phases.shape[0])
        N_curr = min(N, phases.shape[0])
        N_common = min(N_target, N_curr)

        for i in range(N_common):
            delta = jnp.angle(
                jnp.exp(1j * (target_phases[i] - phases[i]))
            )
            active_experts[i].update_omega(delta, lr=self.freq_lr)

        self.modification_log.append({
            "type": "frequency_adaptation",
            "mean_delta": float(jnp.mean(jnp.abs(
                jnp.angle(jnp.exp(1j * (target_phases[:N_common] - phases[:N_common])))
            ))),
        })

    def adapt_siren_weights(self, siren_layer, x, target_phase):
        """Adapt SIREN layer weights via phase alignment.

        Directly nudges weights so that the output phase moves
        toward the target phase. No gradient computation.

        Args:
            siren_layer: PhaseSirenLayer instance.
            x: Input that produced the current phase.
            target_phase: Desired output phase.

        Returns:
            Updated PhaseSirenLayer.
        """
        updated = siren_layer.update_weights_phase(x, target_phase, lr=self.adaptation_lr)
        self.modification_log.append({
            "type": "siren_weight_adaptation",
            "layer_W_norm": float(jnp.linalg.norm(updated.W)),
        })
        return updated

    def adapt_output_projection(self, W, b, expert_output, target_ids, vocab_size, lr=None):
        """Adapt output projection using phase-based weight update.

        Instead of cross-entropy gradient, we use a phase alignment:
          target_logits = one_hot(target_ids)
          current_logits = output_proj @ expert_output + b
          error = target_logits - softmax(current_logits)
          W <- W + lr * expert_output^T @ error
          b <- b + lr * mean(error, axis=0)

        This is technically a gradient step, but it is computed locally
        at the output layer only -- no backprop through the network.
        The rest of the network adapts through phase dynamics.

        Args:
            W: Output projection weights, (phase_dim, vocab_size).
            b: Output bias, (vocab_size,).
            expert_output: Expert combined output, (batch, phase_dim).
            target_ids: Target token IDs, (batch,).
            vocab_size: Vocabulary size.
            lr: Learning rate (uses self.adaptation_lr if None).

        Returns:
            Updated (W, b).
        """
        if lr is None:
            lr = self.adaptation_lr

        # Compute current logits
        logits = jnp.einsum("bi,iv->bv", expert_output, W) + b  # (batch, vocab)
        probs = jax.nn.softmax(logits)
        # Target: one-hot
        target = jax.nn.one_hot(target_ids, vocab_size)  # (batch, vocab)
        # Error signal
        error = target - probs  # (batch, vocab)
        # Local weight update (no backprop)
        dW = lr * jnp.einsum("bi,bv->iv", expert_output, error) / expert_output.shape[0]
        db = lr * jnp.mean(error, axis=0)

        self.modification_log.append({
            "type": "output_projection_adaptation",
            "weight_change_norm": float(jnp.linalg.norm(dW)),
        })
        return W + dW, b + db

    def maybe_clone(self, experts, order_param):
        """Clone an expert if the system is over-synchronized.

        When the order parameter r > clone_threshold, the system
        has lost phase diversity. Cloning the best expert with a
        phase offset restores diversity.

        Args:
            experts: ExpertEnsemble instance.
            order_param: Current Kuramoto order parameter.

        Returns:
            Cloned expert or None.
        """
        cloned = experts.maybe_clone_best(order_param, self.clone_threshold)
        if cloned is not None:
            self.modification_log.append({
                "type": "expert_clone",
                "new_expert_id": cloned.expert_id,
                "parent_phase": float(cloned.phase - 0.2) % (2 * jnp.pi),
                "clone_phase": float(cloned.phase),
                "order_param_at_clone": order_param,
            })
        return cloned

    def maybe_prune(self, experts):
        """Prune experts with low utilization.

        Args:
            experts: ExpertEnsemble instance.
        """
        active_before = experts.num_active
        experts.maybe_prune_worst(self.prune_threshold)
        active_after = experts.num_active
        pruned = active_before - active_after
        if pruned > 0:
            self.modification_log.append({
                "type": "expert_prune",
                "num_pruned": pruned,
                "active_experts_after": active_after,
            })

    def full_adaptation_step(
        self, network, token_ids, target_ids, forward_result
    ):
        """Run all adaptation mechanisms after a forward pass.

        This is the main entry point called by the trainer after
        each batch. It orchestrates:
        1. Coupling adaptation
        2. Frequency adaptation
        3. Output projection adaptation
        4. Expert cloning/pruning

        Args:
            network: PhaseSirenNetwork instance.
            token_ids: Input token IDs, (batch, seq_len).
            target_ids: Target token IDs, (batch, seq_len).
            forward_result: Output from network.forward_pass().

        Returns:
            Dictionary summarizing all modifications made.
        """
        config = network.config
        expert_phases = network.experts.get_phases()
        expert_utils = network.experts.get_omegas()  # Use omegas as proxy for utilization

        # 1. Adapt coupling
        self.adapt_coupling(network.kuramoto, expert_phases, expert_utils)

        # 2. Adapt output projection (local gradient step)
        # Use the last position's logits and targets
        last_logits = forward_result["logits"][:, -1, :]  # (batch, vocab)
        last_targets = target_ids[:, -1]  # (batch,)
        # Get expert output for last position (approximate with phase repr)
        embeddings = network.embed_tokens(token_ids[:, -1])
        phase_repr = network.encode_to_phase(embeddings)
        # Approximate expert output as phase_repr (routed combination)
        network.output_proj_W, network.output_proj_b = self.adapt_output_projection(
            network.output_proj_W, network.output_proj_b,
            phase_repr, last_targets, config.vocab_size
        )

        # 3. Frequency adaptation based on prediction accuracy
        correct = forward_result["correct_per_expert"]
        total = forward_result["total_routed"]
        accuracy = correct / (total + 1e-8)
        # Target phase: nudge experts with low accuracy
        target_phases = expert_phases + (1.0 - accuracy[:, None] if accuracy.ndim > 1 else accuracy)[:, None] * 0.1 * jnp.sign(jnp.sin(expert_phases))
        if target_phases.shape[0] == expert_phases.shape[0]:
            self.adapt_frequencies(network.experts, expert_phases, target_phases)

        # 4. Cloning and pruning
        order_param = forward_result["avg_order_param"]
        self.maybe_clone(network.experts, order_param)
        self.maybe_prune(network.experts)

        return {
            "modifications_this_step": len(self.modification_log),
            "last_modification": self.modification_log[-1] if self.modification_log else None,
        }