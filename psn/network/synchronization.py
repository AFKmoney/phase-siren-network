"""Inter-Instance Synchronization — coupling separate PSN instances.

One of the most novel aspects of the PSN paradigm is that multiple
instances can synchronize their phase dynamics, enabling a form of
"distributed thinking" without explicit message passing.

Mechanism:
  Instance A and Instance B each run their own Kuramoto dynamics.
  Periodically, we compute the cross-instance order parameter:
    r_AB = |(1/N) * sum_i exp(i * (theta_A_i - theta_B_i))|
  If r_AB < 1 (instances are not perfectly synced), we apply
  a weak coupling kick:
    theta_A_i <- theta_A_i + eta * sin(theta_B_i - theta_A_i)
    theta_B_i <- theta_B_i + eta * sin(theta_A_i - theta_B_i)

This is exactly the Kuramoto coupling applied across instances.
The synchronization strength eta controls how quickly instances
converge to a shared phase state.

Properties:
- If instances process similar data, they naturally synchronize
- If instances process different data, they maintain separate attractors
- Synchronization creates implicit knowledge transfer
- No gradients or explicit communication needed
"""
import jax
import jax.numpy as jnp
import math
from typing import Dict, Any, Tuple, Optional


def cross_instance_order_parameter(
    phases_A: jax.Array, phases_B: jax.Array
) -> Tuple[float, float]:
    """Compute the cross-instance synchronization order parameter.

    r_AB * exp(i * psi_AB) = (1/N) * sum_i exp(i * (theta_A_i - theta_B_i))

    Args:
        phases_A: Phases of instance A, shape (N,).
        phases_B: Phases of instance B, shape (N,).

    Returns:
        r_AB: Cross-instance coherence in [0, 1].
        psi_AB: Cross-instance mean phase difference in [0, 2*pi).
    """
    # Handle mismatched sizes by truncating to min
    N = min(phases_A.shape[0], phases_B.shape[0])
    z = jnp.mean(jnp.exp(1j * (phases_A[:N] - phases_B[:N])))
    r = float(jnp.abs(z))
    psi = float(jnp.angle(z)) % (2 * math.pi)
    return r, psi


class InterInstanceSynchronizer:
    """Manages synchronization between multiple PSN instances.

    Each instance has its own Kuramoto system. The synchronizer
    periodically applies weak coupling between instances to allow
    knowledge transfer through phase alignment.

    Attributes:
        sync_coupling: Coupling strength between instances.
        sync_interval: Synchronize every N tokens.
        num_instances: Number of managed instances.
        phase_histories: List of phase histories for each instance.
    """

    def __init__(self, sync_coupling: float = 0.1, sync_interval: int = 50):
        self.sync_coupling = sync_coupling
        self.sync_interval = sync_interval
        self.phase_histories: Dict[int, jax.Array] = {}
        self.sync_history: list = []
        self.token_count = 0

    def register_instance(self, instance_id: int, phases: jax.Array):
        """Register a PSN instance with its current phases.

        Args:
            instance_id: Unique identifier for the instance.
            phases: Current expert phases, shape (N,).
        """
        self.phase_histories[instance_id] = phases.copy()

    def should_sync(self) -> bool:
        """Check if it's time to synchronize.

        Returns:
            True if sync_interval tokens have been processed since last sync.
        """
        return self.token_count > 0 and self.token_count % self.sync_interval == 0

    def synchronize_pair(
        self,
        phases_A: jax.Array,
        phases_B: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, float]:
        """Synchronize two instances by applying cross-coupling.

        Each instance's phases are nudged toward the other's:
          theta_A_i <- theta_A_i + eta * sin(theta_B_i - theta_A_i)
          theta_B_i <- theta_B_i + eta * sin(theta_A_i - theta_B_i)

        Args:
            phases_A: Phases of instance A, shape (N,).
            phases_B: Phases of instance B, shape (N,).

        Returns:
            new_phases_A: Synchronized phases for instance A.
            new_phases_B: Synchronized phases for instance B.
            r_AB: Cross-instance order parameter before sync.
        """
        N = min(phases_A.shape[0], phases_B.shape[0])
        r_AB, _ = cross_instance_order_parameter(phases_A, phases_B)

        # Cross-coupling kicks
        kick_A = self.sync_coupling * jnp.sin(phases_B[:N] - phases_A[:N])
        kick_B = self.sync_coupling * jnp.sin(phases_A[:N] - phases_B[:N])

        new_phases_A = phases_A.at[:N].add(kick_A)
        new_phases_B = phases_B.at[:N].add(kick_B)

        # Wrap to [0, 2*pi)
        new_phases_A = new_phases_A % (2 * jnp.pi)
        new_phases_B = new_phases_B % (2 * jnp.pi)

        self.sync_history.append({
            "token_count": self.token_count,
            "r_AB": r_AB,
        })

        return new_phases_A, new_phases_B, r_AB

    def synchronize_all(self) -> Dict[int, jax.Array]:
        """Synchronize all registered instances pairwise.

        For M instances, performs M*(M-1)/2 pairwise synchronizations.
        The order of pairwise sync matters for convergence but not
        for the final equilibrium (which is the same for any order).

        Returns:
            Dictionary mapping instance_id to synchronized phases.
        """
        instance_ids = list(self.phase_histories.keys())
        if len(instance_ids) < 2:
            return dict(self.phase_histories)

        # Collect all pairwise syncs
        updated = dict(self.phase_histories)
        for i in range(len(instance_ids)):
            for j in range(i + 1, len(instance_ids)):
                id_A, id_B = instance_ids[i], instance_ids[j]
                new_A, new_B, _ = self.synchronize_pair(
                    updated[id_A], updated[id_B]
                )
                updated[id_A] = new_A
                updated[id_B] = new_B

        self.phase_histories = updated
        return updated

    def update_token_count(self, count: int):
        """Update the token counter (call after each batch)."""
        self.token_count = count

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics.

        Returns:
            Dictionary with sync history and current state.
        """
        return {
            "num_instances": len(self.phase_histories),
            "total_syncs": len(self.sync_history),
            "last_sync_r": self.sync_history[-1]["r_AB"] if self.sync_history else 0.0,
            "sync_history": self.sync_history[-10:],  # Last 10 syncs
        }