"""
PSN Configuration — all hyperparameters for the Phase Siren Network.

Every architectural knob is centralized here so experiments are reproducible
and the whitepaper can reference concrete parameter values.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PSNConfig:
    """Full configuration for a Phase Siren Network."""

    # ------------------------------------------------------------------
    # Token / Embedding
    # ------------------------------------------------------------------
    vocab_size: int = 67           # Shakespeare char-level: 26 letters + 10 digits + punctuation + space
    embedding_dim: int = 64        # Dimensionality of token embeddings before phase encoding

    # ------------------------------------------------------------------
    # Phase Siren Layer
    # ------------------------------------------------------------------
    omega_0: float = 30.0          # Base angular frequency for SIREN (Sitzmann et al. 2020)
    phase_dim: int = 128           # Number of phase oscillators per SIREN layer
    siren_hidden_dim: int = 256    # Hidden dimension within SIREN projections

    # ------------------------------------------------------------------
    # Kuramoto Attractor
    # ------------------------------------------------------------------
    num_experts: int = 8           # Number of expert oscillators
    kuramoto_K: float = 4.0        # Global coupling strength
    dt: float = 0.01               # Integration time step
    num_ticks: int = 10            # Number of phase ticks per token (continuous processing)
    rk4_substeps: int = 4          # RK4 sub-steps per tick (fixed at 4 for classical RK4)

    # ------------------------------------------------------------------
    # Phase Hash Router
    # ------------------------------------------------------------------
    hash_dim: int = 64             # Dimensionality of the hash projection space
    routing_top_k: int = 2         # Number of experts each token is routed to
    hash_temperature: float = 1.0  # Softmax temperature for routing weights

    # ------------------------------------------------------------------
    # Self-Modification
    # ------------------------------------------------------------------
    self_mod_lr: float = 0.01      # Learning rate for phase-based parameter adaptation
    clone_threshold: float = 0.95  # Order parameter threshold to trigger expert cloning
    prune_threshold: float = 0.05  # Minimum utilization below which an expert is pruned
    max_experts: int = 32          # Hard cap on expert count (prevents unbounded growth)

    # ------------------------------------------------------------------
    # Synchronization (inter-instance)
    # ------------------------------------------------------------------
    sync_coupling: float = 0.1     # Coupling strength between instances
    sync_interval: int = 50        # Synchronize every N tokens

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    seq_len: int = 256             # Context window length
    batch_size: int = 4            # Number of sequences per batch
    num_epochs: int = 50           # Number of passes over the dataset
    adaptation_lr: float = 0.005   # Phase adaptation rate (replaces gradient LR)
    freq_adaptation: float = 0.001 # Natural frequency adaptation rate

    # ------------------------------------------------------------------
    # Shakespeare-specific
    # ------------------------------------------------------------------
    dataset_path: str = "shakespeare.txt"
    gen_seed: str = "HAMLET "       # Seed text for generation
    gen_length: int = 500          # Number of characters to generate

    # ------------------------------------------------------------------
    # Logging / Metrics
    # ------------------------------------------------------------------
    log_interval: int = 10         # Log every N batches
    save_metrics_path: str = "download/psn_metrics.jsonl"


@dataclass
class ExpertState:
    """Mutable state for a single expert oscillator."""
    phase: float = 0.0             # Current phase theta in [0, 2*pi)
    omega: float = 1.0             # Natural frequency
    utilization: float = 0.0       # Running count of tokens routed to this expert
    correct_count: float = 0.0    # Running count of correct predictions


@dataclass
class PSNState:
    """Full mutable state of a PSN instance."""
    expert_phases: Optional[list] = None   # shape (num_experts,) — current phases
    expert_omegas: Optional[list] = None   # shape (num_experts,) — natural frequencies
    coupling_matrix: Optional[list] = None # shape (num_experts, num_experts) — pairwise coupling
    expert_utils: Optional[list] = None    # shape (num_experts,) — utilization counts
    expert_correct: Optional[list] = None  # shape (num_experts,) — correct prediction counts
    tick_count: int = 0                     # Global tick counter (continuous clock)
    active_experts: int = 0                 # Number of currently active experts
