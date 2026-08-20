"""
Expert Module — a phase-coupled processing unit.

Each expert is an oscillator in the Kuramoto ensemble. Its "computation"
is a function of its phase state and the input it receives. The expert:

1. Receives routed tokens (via phase-resonance routing)
2. Processes them through a small SIREN network (expert-specific)
3. Produces an output that is a function of both the input and its phase
4. Has its phase updated by the Kuramoto attractor (coupling with other experts)
5. Can be cloned, pruned, or have its parameters self-modified

The expert is NOT a standard feedforward network. It is a dynamical
system whose behavior depends on its oscillatory state.

Critical insight: the same expert, receiving the same input, will produce
different outputs at different times because its phase has evolved.
This is the key property that enables temporal reasoning without
explicit recurrence or attention.
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Optional
from .phase_siren import PhaseSirenLayer


class ExpertModule:
    """A single expert oscillator with its own SIREN processor.

    Each expert maintains:
    - A phase theta on the unit circle (from Kuramoto dynamics)
    - A natural frequency omega
    - A small SIREN network for processing routed tokens
    - Utilization and accuracy statistics for self-modification

    Attributes:
        expert_id: Unique identifier for this expert.
        phase: Current phase theta in [0, 2*pi).
        omega: Natural frequency (rad/tick).
        input_dim: Dimension of routed token features.
        output_dim: Dimension of expert output.
        siren: Expert-specific SIREN processing layer.
        utilization: Running count of tokens processed.
        correct_count: Running count of correct predictions.
        active: Whether this expert is currently active.
    """

    def __init__(
        self,
        expert_id: int,
        input_dim: int,
        output_dim: int,
        omega: float = 1.0,
        omega_0: float = 30.0,
        key: jax.Array = None,
    ):
        self.expert_id = expert_id
        self.phase = jnp.array(0.0)
        self.omega = jnp.array(omega)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.utilization = jnp.array(0.0)
        self.correct_count = jnp.array(0.0)
        self.active = True

        if key is None:
            key = random.PRNGKey(expert_id * 1000)
        # Each expert has a small SIREN for processing
        self.siren = PhaseSirenLayer(input_dim, output_dim, omega_0, key)

    def process(self, x: jax.Array) -> Dict[str, jax.Array]:
        """Process routed input through the expert's SIREN, modulated by phase.

        The expert's output is phase-modulated: the SIREN activation is
        multiplied by cos(theta - phi_output), where theta is the
        expert's Kuramoto phase and phi_output is the output phase.
        This means the same input produces different outputs depending
        on when (in the oscillation cycle) the expert processes it.

        Args:
            x: Routed token features, shape (batch, input_dim).

        Returns:
            Dictionary with:
                'output': Phase-modulated expert output, shape (batch, output_dim).
                'phase': Expert's output phase, shape (batch, output_dim).
                'modulation': Phase modulation factor, shape (batch, output_dim).
        """
        activation, phase, _ = self.siren(x)
        # Phase modulation: modulate output by resonance with expert's Kuramoto phase
        # cos(theta_expert - phi_output) is maximal when they are in sync
        modulation = jnp.cos(self.phase - jnp.mean(phase, axis=-1, keepdims=True))
        output = activation * modulation
        return {
            "output": output,
            "phase": phase,
            "modulation": modulation,
        }

    def update_phase(self, new_phase: jax.Array):
        """Update the expert's phase (called by Kuramoto attractor)."""
        self.phase = new_phase

    def update_omega(self, delta_omega: jax.Array, lr: float = 0.001):
        """Adapt natural frequency based on task feedback.

        If the expert is consistently correct, its frequency is left alone.
        If incorrect, omega is nudged to change the expert's temporal
        processing characteristics.

        Args:
            delta_omega: Phase error signal, scalar.
            lr: Frequency adaptation rate.
        """
        self.omega = float(self.omega + lr * float(jnp.sum(delta_omega)))
        # Keep omega positive and bounded
        self.omega = max(0.1, min(10.0, self.omega))

    def record_utilization(self, batch_size: int):
        """Record that this expert processed batch_size tokens."""
        self.utilization = self.utilization + batch_size

    def record_correct(self, count: float):
        """Record correct predictions."""
        self.correct_count = self.correct_count + count

    def get_accuracy(self) -> float:
        """Get this expert's prediction accuracy."""
        if self.utilization < 1e-6:
            return 0.0
        return float(self.correct_count / self.utilization)

    def clone(self, phase_offset: float = 0.1, new_id: Optional[int] = None) -> "ExpertModule":
        """Clone this expert with a phase offset.

        The clone inherits all SIREN weights but starts at a different
        phase. This allows the network to create specialized copies
        of high-performing experts that explore nearby phase regions.

        Args:
            phase_offset: Phase offset for the clone, in radians.
            new_id: ID for the cloned expert.

        Returns:
            New ExpertModule with same weights, offset phase.
        """
        if new_id is None:
            new_id = self.expert_id + 1000
        cloned = ExpertModule(
            expert_id=new_id,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            omega=float(self.omega),
            omega_0=self.siren.omega_0,
            key=random.PRNGKey(new_id),
        )
        # Copy SIREN weights
        cloned.siren.W = self.siren.W.copy()
        cloned.siren.b = self.siren.b.copy()
        # Phase offset
        cloned.phase = (self.phase + phase_offset) % (2 * jnp.pi)
        return cloned


class ExpertEnsemble:
    """Manages a collection of expert oscillators.

    The ensemble handles:
    - Creating/initializing experts
    - Cloning high-performing experts
    - Pruning low-utilization experts
    - Aggregating expert outputs via routing weights

    Attributes:
        experts: List of ExpertModule instances.
        max_experts: Maximum number of allowed experts.
    """

    def __init__(self, num_experts: int, input_dim: int, output_dim: int,
                 omega_0: float = 30.0, key: jax.Array = None):
        if key is None:
            key = random.PRNGKey(999)
        self.max_experts = 32
        self.experts = []
        for i in range(num_experts):
            k_i = random.fold_in(key, i)
            # Spread initial natural frequencies for diversity
            omega = 0.5 + 2.0 * i / num_experts
            self.experts.append(
                ExpertModule(i, input_dim, output_dim, omega, omega_0, k_i)
            )

    @property
    def num_active(self) -> int:
        return sum(1 for e in self.experts if e.active)

    def get_phases(self) -> jax.Array:
        """Get current phases of all active experts.

        Returns:
            Phase array of shape (num_active,).
        """
        return jnp.array([e.phase for e in self.experts if e.active])

    def get_omegas(self) -> jax.Array:
        """Get natural frequencies of all active experts.

        Returns:
            Omega array of shape (num_active,).
        """
        return jnp.array([e.omega for e in self.experts if e.active])

    def set_phases(self, phases: jax.Array):
        """Update phases of all active experts (from Kuramoto step)."""
        idx = 0
        for e in self.experts:
            if e.active:
                e.update_phase(phases[idx])
                idx += 1

    def process_with_routing(
        self, x: jax.Array, routing_weights: jax.Array, top_k_indices: jax.Array
    ) -> jax.Array:
        """Process input through routed experts and combine outputs.

        Args:
            x: Input features, shape (batch, input_dim).
            routing_weights: Routing weights, shape (batch, num_active).
            top_k_indices: Selected expert indices, shape (batch, top_k).

        Returns:
            Combined output, shape (batch, output_dim).
        """
        batch_size = x.shape[0]
        active_experts = [e for e in self.experts if e.active]
        output_dim = active_experts[0].output_dim

        # Weighted sum of expert outputs
        combined = jnp.zeros((batch_size, output_dim))

        for local_idx, expert in enumerate(active_experts):
            # Get routing weight for this expert across the batch
            weights = routing_weights[:, local_idx]  # (batch,)
            # Only process if any token routes here
            if jnp.any(weights > 1e-6):
                expert_out = expert.process(x)
                combined = combined + weights[:, None] * expert_out["output"]
                # Record utilization
                expert.record_utilization(int(jnp.sum(weights > 1e-6)))

        return combined

    def maybe_clone_best(self, order_param: float, clone_threshold: float = 0.95
    ) -> Optional["ExpertModule"]:
        """Clone the best expert if order parameter is very high.

        When the system is too synchronized (order parameter near 1),
        diversity collapses. Cloning the best expert with a phase
        offset introduces controlled desynchronization.

        Args:
            order_param: Kuramoto order parameter r in [0, 1].
            clone_threshold: Trigger cloning when r > this value.

        Returns:
            Newly cloned expert, or None if no cloning occurred.
        """
        if order_param < clone_threshold:
            return None
        if self.num_active >= self.max_experts:
            return None
        # Find the most accurate expert
        best = max(self.experts, key=lambda e: e.get_accuracy() if e.active else 0)
        if best.get_accuracy() < 0.3:
            return None
        # Clone with a small phase offset
        new_id = max(e.expert_id for e in self.experts) + 1
        cloned = best.clone(phase_offset=0.2, new_id=new_id)
        self.experts.append(cloned)
        return cloned

    def maybe_prune_worst(self, prune_threshold: float = 0.05):
        """Prune experts with very low utilization.

        An expert that never gets routed to is wasting computational
        budget. We deactivate (not delete) it so it can be
        re-activated if needed.

        Args:
            prune_threshold: Deactivate if utilization < this fraction
                of the total.
        """
        total_util = sum(e.utilization for e in self.experts if e.active)
        if total_util < 1e-6:
            return
        for e in self.experts:
            if e.active and e.utilization / total_util < prune_threshold:
                e.active = False

    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about all experts.

        Returns:
            Dictionary with utilization, accuracy, phase info per expert.
        """
        stats = []
        for e in self.experts:
            stats.append({
                "id": e.expert_id,
                "active": e.active,
                "phase": float(jnp.sum(e.phase)),
                "omega": float(jnp.sum(e.omega)) if hasattr(e.omega, 'ndim') else float(e.omega),
                "utilization": float(jnp.sum(e.utilization)),
                "accuracy": e.get_accuracy(),
            })
        return stats
