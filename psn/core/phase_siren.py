"""Phase Siren Layer — sinusoidal activation with phase-encoded representations.

Traditional SIREN (Sitzmann et al. 2020) uses sin(omega_0 * Wx + b) as an
activation function, producing high-frequency signal representations.

We extend this into a **phase dynamics** framework:
- Input tokens are projected into phase space: phi = W @ x + b
- The phase phi is NOT a static embedding — it evolves over time via Kuramoto coupling
- Information is encoded in the **angular position** on the unit circle, not amplitude
- This makes every neuron a rotating oscillator, not a scalar unit

Mathematical formulation:
  Given input x in R^d, embedding dimension m:
    phi_i = sum_j W_ij * x_j + b_i   (linear projection to phase space)
    output_i = sin(omega_0 * phi_i)   (sinusoidal activation)
    phase_i = mod(omega_0 * phi_i, 2*pi)  (extract phase for Kuramoto dynamics)

The phase extraction mod(phi, 2*pi) is what feeds into the Kuramoto attractor.
The sin() output is what feeds into the next layer's computation.
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any


class PhaseSirenLayer:
    """A single SIREN layer that produces phase-encoded activations.

    Unlike standard SIREN, we explicitly track and return the **phase**
    of each neuron, not just the sinusoidal activation. This phase is
    the fundamental currency of the PSN — it flows into Kuramoto coupling,
    hash routing, and self-modification.

    Attributes:
        input_dim: Dimensionality of input features.
        output_dim: Number of phase oscillators (neurons) in this layer.
        omega_0: Base angular frequency controlling the spectral bandwidth.
        W: Weight matrix projecting input to phase space.
        b: Bias vector (phase offsets).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 30.0,
        key: jax.Array = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        # Initialize weights with SIREN-specific first-layer initialization:
        # W ~ Uniform(-1/sqrt(input_dim), 1/sqrt(input_dim))
        # This ensures the first layer preserves input frequencies
        if key is None:
            key = random.PRNGKey(42)
        k1, k2 = random.split(key)
        self.W = random.uniform(
            k1, (input_dim, output_dim),
            minval=-1.0 / jnp.sqrt(input_dim),
            maxval=1.0 / jnp.sqrt(input_dim),
        )
        self.b = random.uniform(k2, (output_dim,), minval=0.0, maxval=2 * jnp.pi)

    def __call__(
        self, x: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass: compute sinusoidal activation and extract phase.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            activation: sin(omega_0 * phi) of shape (..., output_dim).
                This is the SIREN activation that flows to the next layer.
            phase: mod(omega_0 * phi, 2*pi) of shape (..., output_dim).
                This is the phase that feeds into Kuramoto dynamics.
            phi_raw: The pre-modulation projection of shape (..., output_dim).
                Useful for analysis and debugging.
        """
        # Linear projection into phase space
        phi = jnp.einsum("...i,ij->...j", x, self.W) + self.b  # (..., output_dim)

        # Apply SIREN frequency scaling
        phi_scaled = self.omega_0 * phi  # (..., output_dim)

        # Sinusoidal activation (standard SIREN output)
        activation = jnp.sin(phi_scaled)  # (..., output_dim)

        # Extract phase on the unit circle [0, 2*pi)
        phase = jnp.mod(phi_scaled, 2 * jnp.pi)  # (..., output_dim)

        return activation, phase, phi_scaled

    def get_phase_fingerprint(self, x: jax.Array) -> jax.Array:
        """Compute a compact phase fingerprint for routing.

        Reduces the full phase vector to a single scalar phase by
        computing the circular mean — the angle of the mean vector
        on the unit circle.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Scalar phase fingerprint in [0, 2*pi), shape (...).
        """
        _, phase, _ = self(x)
        # Circular mean: angle of the sum of unit complex phasors
        z = jnp.exp(1j * phase)  # (..., output_dim)
        mean_z = jnp.mean(z, axis=-1)  # (...)
        fingerprint = jnp.angle(mean_z) % (2 * jnp.pi)  # (...) in [0, 2pi)
        return fingerprint

    def update_weights_phase(self, x: jax.Array, target_phase: jax.Array, lr: float = 0.01
    ) -> "PhaseSirenLayer":
        """Adapt weights via phase alignment (non-gradient).

        Instead of backpropagation, we nudge W and b so that the
        output phase moves toward target_phase. This is a direct
        phase-space manipulation, not a gradient step.

        Adaptation rule:
          delta_phi = target_phase - current_phase  (circular difference)
          W <- W + lr * delta_phi @ x^T  (outer product nudge)
          b <- b + lr * delta_phi        (bias nudge)

        Args:
            x: Input of shape (batch, input_dim).
            target_phase: Desired phase of shape (batch, output_dim).
            lr: Phase adaptation learning rate.

        Returns:
            New PhaseSirenLayer with updated weights.
        """
        _, current_phase, _ = self(x)
        # Circular phase difference: shortest arc on the unit circle
        delta = jnp.angle(jnp.exp(1j * (target_phase - current_phase)))
        # Weight update via outer product (phase-aligned, not gradient-aligned)
        dW = lr * jnp.einsum("bi,bj->ij", delta, x) / x.shape[0]
        db = lr * jnp.mean(delta, axis=0)
        new_layer = PhaseSirenLayer(
            self.input_dim, self.output_dim, self.omega_0, key=None
        )
        new_layer.W = self.W + dW
        new_layer.b = self.b + db
        return new_layer


class PhaseSirenStack:
    """Stacked Phase Siren layers — a deep phase-encoded network.

    Each layer takes the sinusoidal activation of the previous layer
    and produces new phase-encoded representations. The phase flows
    through the entire stack and is extracted at each level for
    Kuramoto coupling.

    Architecture:
        x -> [SIREN_1] -> (act_1, phase_1)
           -> [SIREN_2] -> (act_2, phase_2)
           -> ...
           -> [SIREN_L] -> (act_L, phase_L)

    All intermediate phases are collected for multi-scale Kuramoto coupling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        omega_0: float = 30.0,
        key: jax.Array = None,
    ):
        if key is None:
            key = random.PRNGKey(42)
        self.layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            k_i = random.fold_in(key, i)
            self.layers.append(PhaseSirenLayer(dims[i], dims[i + 1], omega_0, k_i))
        self.num_layers = num_layers

    def __call__(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """Forward pass through all SIREN layers.

        Args:
            x: Input of shape (batch, input_dim).

        Returns:
            Dictionary with:
                'output': Final activation, shape (batch, output_dim).
                'phases': All layer phases, list of (batch, dim_i) arrays.
                'activations': All layer activations, list of (batch, dim_i) arrays.
        """
        phases = []
        activations = []
        h = x
        for layer in self.layers:
            h, phase, _ = layer(h)
            activations.append(h)
            phases.append(phase)
        return {
            "output": h,
            "phases": phases,
            "activations": activations,
        }

    def get_final_phase(self, x: jax.Array) -> jax.Array:
        """Get the phase from the last SIREN layer.

        Args:
            x: Input of shape (batch, input_dim).

        Returns:
            Phase of shape (batch, last_layer_dim).
        """
        result = self(x)
        return result["phases"][-1]
