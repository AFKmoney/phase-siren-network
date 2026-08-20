"""Kuramoto Attractor — RK4-integrated coupled oscillator dynamics.

The Kuramoto model (Kuramoto, 1984) describes N coupled phase oscillators:

    d(theta_i)/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)

where:
- theta_i is the phase of oscillator i
- omega_i is its natural frequency
- K is the global coupling strength
- N is the number of oscillators

Key properties:
1. For K > K_critical, oscillators spontaneously synchronize
2. The order parameter r*e^(i*psi) = (1/N)*sum_j e^(i*theta_j) measures coherence
3. r=0 means incoherent, r=1 means fully synchronized

We use classical 4th-order Runge-Kutta (RK4) for numerical integration,
which provides O(dt^4) local error — critical for maintaining long-term
stability of the phase dynamics.

The Kuramoto attractor serves as the "communication backbone" of the PSN:
- Experts that process related tokens naturally synchronize
- Synchronization creates implicit communication channels
- The coupling matrix K can be made non-uniform for structured expertise
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any, Optional
import math


def kuramoto_rhs(
    phases: jax.Array,
    omegas: jax.Array,
    coupling_matrix: jax.Array,
    K_global: float,
) -> jax.Array:
    """Compute the right-hand side of the Kuramoto ODE system.

    d(theta_i)/dt = omega_i + (K_global/N) * sum_j K_ij * sin(theta_j - theta_i)

    When coupling_matrix is uniform (all ones), this reduces to the
    standard Kuramoto model. Non-uniform coupling allows for structured
    expert specialization.

    Args:
        phases: Current phases, shape (N,).
        omegas: Natural frequencies, shape (N,).
        coupling_matrix: Pairwise coupling strengths, shape (N, N).
            K_ij controls how strongly oscillator j influences i.
        K_global: Global coupling strength multiplier.

    Returns:
        Phase derivatives d(theta)/dt, shape (N,).
    """
    N = phases.shape[0]
    # Phase differences: theta_j - theta_i for all pairs
    # (N, 1) - (1, N) -> (N, N)
    phase_diff = phases[None, :] - phases[:, None]
    # Coupling term: K_ij * sin(theta_j - theta_i)
    coupling_term = coupling_matrix * jnp.sin(phase_diff)  # (N, N)
    # Sum over j (axis=1), normalized by N and scaled by K_global
    coupling_sum = K_global / N * jnp.sum(coupling_term, axis=1)  # (N,)
    # Full derivative: natural frequency + coupling
    dtheta_dt = omegas + coupling_sum  # (N,)
    return dtheta_dt


def rk4_step(
    phases: jax.Array,
    omegas: jax.Array,
    coupling_matrix: jax.Array,
    K_global: float,
    dt: float,
) -> jax.Array:
    """Perform one RK4 integration step for the Kuramoto system.

    The classical RK4 method:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y(t+dt) = y(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Note: the Kuramoto system is autonomous (no explicit t dependence),
    so we omit t from the function arguments.

    Args:
        phases: Current phases, shape (N,).
        omegas: Natural frequencies, shape (N,).
        coupling_matrix: Pairwise coupling, shape (N, N).
        K_global: Global coupling strength.
        dt: Time step.

    Returns:
        New phases after one RK4 step, shape (N,).
    """
    k1 = kuramoto_rhs(phases, omegas, coupling_matrix, K_global)
    k2 = kuramoto_rhs(phases + 0.5 * dt * k1, omegas, coupling_matrix, K_global)
    k3 = kuramoto_rhs(phases + 0.5 * dt * k2, omegas, coupling_matrix, K_global)
    k4 = kuramoto_rhs(phases + dt * k3, omegas, coupling_matrix, K_global)
    new_phases = phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Wrap to [0, 2*pi)
    new_phases = new_phases % (2 * jnp.pi)
    return new_phases


def compute_order_parameter(phases: jax.Array) -> Tuple[float, float]:
    """Compute the Kuramoto order parameter r and mean phase psi.

    r * e^(i*psi) = (1/N) * sum_j e^(i*theta_j)

    r in [0, 1] measures synchronization:
    - r ~ 0: incoherent (phases uniformly distributed)
    - r ~ 1: fully synchronized (all phases aligned)

    Args:
        phases: Phase array, shape (N,).

    Returns:
        r: Order parameter magnitude in [0, 1].
        psi: Mean phase in [0, 2*pi).
    """
    z = jnp.mean(jnp.exp(1j * phases))
    r = float(jnp.abs(z))
    psi = float(jnp.angle(z)) % (2 * math.pi)
    return r, psi


class KuramotoAttractor:
    """Manages the Kuramoto dynamics for a set of expert oscillators.

    The attractor wraps the RK4 integrator and provides:
    - Multi-step ticking (simulate multiple dt steps per token)
    - Dynamic coupling strength adaptation
    - Phase perturbation from external inputs (token-driven)
    - Order parameter monitoring

    Attributes:
        num_oscillators: Current number of active oscillators.
        K: Global coupling strength.
        dt: Integration time step.
        coupling_matrix: Pairwise coupling matrix, shape (N, N).
    """

    def __init__(
        self,
        num_oscillators: int,
        K: float = 4.0,
        dt: float = 0.01,
        heterogeneous: bool = True,
        key: jax.Array = None,
    ):
        self.num_oscillators = num_oscillators
        self.K = K
        self.dt = dt

        if key is None:
            key = random.PRNGKey(42)
        # Coupling matrix: start with uniform all-to-all coupling
        self.coupling_matrix = jnp.ones((num_oscillators, num_oscillators))
        # No self-coupling (diagonal = 0)
        self.coupling_matrix = self.coupling_matrix.at[jnp.diag_indices(num_oscillators)].set(0.0)

    def tick(
        self,
        phases: jax.Array,
        omegas: jax.Array,
        num_steps: int = 1,
        external_kick: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """Advance the Kuramoto system by num_steps RK4 steps.

        Optionally applies an external phase kick from token processing.
        This couples the symbolic processing (SIREN + routing) with
        the dynamical substrate (Kuramoto oscillators).

        Args:
            phases: Current phases, shape (N,).
            omegas: Natural frequencies, shape (N,).
            num_steps: Number of RK4 integration steps.
            external_kick: Optional phase perturbation from tokens, shape (N,).

        Returns:
            new_phases: Phases after integration, shape (N,).
            diagnostics: Dictionary with order parameter history, etc.
        """
        current_phases = phases.copy()
        r_history = []

        for step in range(num_steps):
            current_phases = rk4_step(
                current_phases, omegas, self.coupling_matrix, self.K, self.dt
            )
            # Apply external kick if provided
            if external_kick is not None:
                current_phases = (current_phases + external_kick) % (2 * jnp.pi)
            r, psi = compute_order_parameter(current_phases)
            r_history.append(r)

        return current_phases, {
            "order_parameter": r_history[-1] if r_history else 0.0,
            "mean_phase": psi if r_history else 0.0,
            "r_history": r_history,
        }

    def adapt_coupling(
        self,
        phases: jax.Array,
        expert_utils: jax.Array,
        lr: float = 0.01,
    ) -> jax.Array:
        """Adapt the coupling matrix based on expert utilization and phase alignment.

        Hebbian-like rule for coupling adaptation:
        - If two experts are often used together AND are phase-aligned,
          strengthen their coupling (K_ij increases).
        - If they are phase-anti-aligned, weaken coupling.

        K_ij <- K_ij + lr * (util_i * util_j * cos(theta_i - theta_j))

        Args:
            phases: Current phases, shape (N,).
            expert_utils: Utilization counts, shape (N,).
            lr: Coupling adaptation rate.

        Returns:
            Updated coupling matrix, shape (N, N).
        """
        N = phases.shape[0]
        phase_diff = phases[None, :] - phases[:, None]  # (N, N)
        # Hebbian coupling: strengthen co-utilized, phase-aligned experts
        util_outer = expert_utils[None, :] * expert_utils[:, None]  # (N, N)
        dK = lr * util_outer * jnp.cos(phase_diff)  # (N, N)
        # Normalize by max to prevent unbounded growth
        dK = dK / (jnp.max(jnp.abs(dK)) + 1e-8)
        new_coupling = self.coupling_matrix + dK
        # Ensure non-negative coupling and no self-coupling
        new_coupling = jnp.maximum(new_coupling, 0.0)
        new_coupling = new_coupling.at[jnp.diag_indices(N)].set(0.0)
        self.coupling_matrix = new_coupling
        return new_coupling

    def resize(self, new_size: int, key: jax.Array = None) -> "KuramotoAttractor":
        """Resize the coupling matrix when experts are cloned/pruned.

        Args:
            new_size: New number of oscillators.
            key: JAX random key for initializing new coupling rows/columns.

        Returns:
            Self with resized coupling matrix.
        """
        old_size = self.num_oscillators
        if new_size == old_size:
            return self
        if key is None:
            key = random.PRNGKey(0)
        if new_size > old_size:
            # Expand: add new rows/columns with default coupling of 1.0
            old_coupling = self.coupling_matrix
            new_coupling = jnp.ones((new_size, new_size))
            new_coupling = new_coupling.at[jnp.diag_indices(new_size)].set(0.0)
            new_coupling = new_coupling.at[:old_size, :old_size].set(old_coupling)
            # New experts couple weakly at first
            k1, k2 = random.split(key)
            new_coupling = new_coupling.at[:old_size, old_size:].set(random.uniform(k1, (old_size, new_size - old_size), 0.1, 0.5))
            new_coupling = new_coupling.at[old_size:, :old_size].set(random.uniform(k2, (new_size - old_size, old_size), 0.1, 0.5))
            self.coupling_matrix = new_coupling
        else:
            # Shrink: take the top-left submatrix
            self.coupling_matrix = self.coupling_matrix[:new_size, :new_size]
        self.num_oscillators = new_size
        return self
