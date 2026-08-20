"""PSN Trainer - orchestrates the training loop.

Unlike standard ML trainers, the PSN trainer does NOT:
- Compute gradients via backpropagation
- Use an optimizer (Adam, SGD, etc.)
- Minimize a loss function through gradient descent

Instead, the trainer:
1. Feeds sequences through the PSN forward pass
2. Collects phase dynamics and prediction accuracy
3. Calls the SelfModifier to apply phase-based adaptations
4. Monitors the Kuramoto order parameter and expert statistics
5. Periodically synchronizes with other instances
6. Logs everything for analysis and visualization

The "training signal" is the prediction accuracy at the output layer.
This accuracy is used by the SelfModifier to adapt:
- Output projection (local gradient step at the output only)
- Expert frequencies (phase alignment rule)
- Coupling matrix (Hebbian phase rule)
- Expert population (cloning/pruning)
- SIREN weights (phase nudge rule)
- Hash router (fingerprint alignment rule)

The key insight: learning emerges from the dynamics, not from
an optimization algorithm. The trainer is a passive observer
that occasionally nudges parameters based on observed behavior.
"""

import jax
import jax.numpy as jnp
import json
import time
import os
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psn.config import PSNConfig
from psn.network.psn import PhaseSirenNetwork
from psn.network.self_modification import SelfModifier
from psn.network.synchronization import InterInstanceSynchronizer
from psn.training.shakespeare import ShakespeareData
from psn.metrics.metrics import compute_all_metrics


class PSNTrainer:
    """Trainer for the Phase Siren Network.

    Manages the non-gradient training loop, logging, and generation.

    Attributes:
        config: PSNConfig hyperparameters.
        network: The PSN instance being trained.
        data: Shakespeare dataset.
        modifier: Self-modification system.
        synchronizer: Inter-instance synchronization.
        metrics_log: List of per-step metric dictionaries.
    """

    def __init__(self, config: PSNConfig = None):
        if config is None:
            config = PSNConfig()
        self.config = config

        print("=" * 60)
        print("Phase Siren Network (PSN) -- Non-Gradient Training")
        print("=" * 60)

        # Load Shakespeare data
        print("\n[Trainer] Loading Shakespeare dataset...")
        self.data = ShakespeareData(seq_len=config.seq_len)

        # Override vocab_size from data
        config.vocab_size = self.data.vocab_size
        print(f"[Trainer] Vocab size set to {config.vocab_size}")

        # Initialize network
        print("\n[Trainer] Initializing PSN...")
        self.network = PhaseSirenNetwork(config)
        print(f"[Trainer] PSN initialized with {config.num_experts} experts")

        # Self-modification system
        self.modifier = SelfModifier(
            adaptation_lr=config.adaptation_lr,
            freq_lr=config.freq_adaptation,
            clone_threshold=config.clone_threshold,
            prune_threshold=config.prune_threshold,
        )

        # Inter-instance synchronizer (for future multi-instance use)
        self.synchronizer = InterInstanceSynchronizer(
            sync_coupling=config.sync_coupling,
            sync_interval=config.sync_interval,
        )

        self.metrics_log: List[Dict[str, Any]] = []
        self.step = 0

    def train_step(self, batch_idx: int) -> Dict[str, Any]:
        """Execute one training step.

        Args:
            batch_idx: Index into the dataset for batch selection.

        Returns:
            Dictionary with all metrics for this step.
        """
        config = self.config

        # Get batch
        input_ids, target_ids = self.data.get_batch(config.batch_size, batch_idx)

        # Forward pass
        forward_result = self.network.forward_pass(input_ids, target_ids)

        # Compute accuracy
        predicted = jnp.argmax(forward_result["logits"], axis=-1)
        accuracy = float(jnp.mean(predicted == target_ids))

        # Compute bits per character (information-theoretic measure)
        logits_flat = forward_result["logits"].reshape(-1, config.vocab_size)
        targets_flat = target_ids.reshape(-1)
        log_probs = jax.nn.log_softmax(logits_flat)
        nll = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
        bits_per_char = float(nll / jnp.log(2.0))

        # Self-modification step (adapt all components)
        mod_result = self.modifier.full_adaptation_step(
            self.network, input_ids, target_ids, forward_result
        )

        # Compute additional metrics
        expert_stats = self.network.experts.get_expert_stats()
        routing_entropy = 0.0
        if forward_result["routing_weights"]:
            last_routing = forward_result["routing_weights"][-1]
            if hasattr(self.network.router, 'routing_entropy'):
                routing_entropy = self.network.router.routing_entropy(last_routing)

        step_metrics = {
            "step": self.step,
            "batch_idx": batch_idx,
            "accuracy": accuracy,
            "bits_per_char": bits_per_char,
            "order_parameter": forward_result["avg_order_param"],
            "routing_entropy": routing_entropy,
            "active_experts": self.network.experts.num_active,
            "tick_count": self.network.tick_count,
            "modifications": mod_result["modifications_this_step"],
            "expert_phases": [e["phase"] for e in expert_stats if e["active"]],
            "expert_omegas": [e["omega"] for e in expert_stats if e["active"]],
            "expert_accuracies": [e["accuracy"] for e in expert_stats if e["active"]],
            "expert_utilizations": [e["utilization"] for e in expert_stats if e["active"]],
        }

        self.metrics_log.append(step_metrics)
        self.step += 1
        self.synchronizer.update_token_count(self.step)

        return step_metrics

    def train(self, num_epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs (uses config if None).

        Returns:
            Complete metrics log.
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        config = self.config
        steps_per_epoch = max(1, self.data.num_sequences // config.batch_size)
        total_steps = num_epochs * steps_per_epoch

        print(f"\n{'=' * 60}")
        print(f"Training: {num_epochs} epochs, {steps_per_epoch} steps/epoch")
        print(f"Total steps: {total_steps}")
        print(f"Batch size: {config.batch_size}, Seq len: {config.seq_len}")
        print(f"Experts: {config.num_experts}, K: {config.kuramoto_K}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_accuracy = []
            epoch_bpc = []
            epoch_order = []

            for s in range(steps_per_epoch):
                batch_idx = (epoch * steps_per_epoch + s) * config.batch_size
                metrics = self.train_step(batch_idx)

                epoch_accuracy.append(metrics["accuracy"])
                epoch_bpc.append(metrics["bits_per_char"])
                epoch_order.append(metrics["order_parameter"])

                # Periodic logging
                if (self.step) % config.log_interval == 0:
                    self._log_progress(metrics, epoch, s, steps_per_epoch)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
            print(f"    Avg accuracy: {jnp.mean(jnp.array(epoch_accuracy)):.4f}")
            print(f"    Avg BPC: {jnp.mean(jnp.array(epoch_bpc)):.4f}")
            print(f"    Avg order param: {jnp.mean(jnp.array(epoch_order)):.4f}")
            print(f"    Active experts: {self.network.experts.num_active}")

            # Generate sample text every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._generate_and_print()

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Final accuracy: {self.metrics_log[-1]['accuracy']:.4f}")
        print(f"Final BPC: {self.metrics_log[-1]['bits_per_char']:.4f}")
        print(f"{'=' * 60}")

        # Save metrics
        self._save_metrics()

        return self.metrics_log

    def _log_progress(self, metrics, epoch, step, steps_per_epoch):
        """Print progress during training."""
        print(
            f"  Step {self.step:5d} | "
            f"Epoch {epoch + 1} [{step + 1:3d}/{steps_per_epoch}] | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"BPC: {metrics['bits_per_char']:.4f} | "
            f"r: {metrics['order_parameter']:.3f} | "
            f"Experts: {metrics['active_experts']} | "
            f"Mods: {metrics['modifications']}"
        )

    def _generate_and_print(self):
        """Generate and print a sample text."""
        seed = self.config.gen_seed
        seed_ids = self.data.encode(seed)
        generated_ids = self.network.generate(seed_ids, self.config.gen_length)
        generated_text = self.data.decode(generated_ids)
        print(f"\n  --- Generated Text ---")
        print(f"  {generated_text[:200]}...")
        print(f"  ----------------------")

    def _save_metrics(self):
        """Save metrics log to JSONL file."""
        path = self.config.save_metrics_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for m in self.metrics_log:
                # Convert jax arrays to lists for JSON serialization
                serializable = {}
                for k, v in m.items():
                    if isinstance(v, (list,)):
                        serializable[k] = [float(x) for x in v]
                    elif hasattr(v, 'item'):
                        serializable[k] = float(v)
                    else:
                        serializable[k] = v
                f.write(json.dumps(serializable) + "\n")
        print(f"\n[Trainer] Metrics saved to {path}")

    def generate_text(self, seed: str = None, length: int = None, temperature: float = 0.8
    ) -> str:
        """Generate text from the trained network.

        Args:
            seed: Seed text (uses config default if None).
            length: Number of characters to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        if seed is None:
            seed = self.config.gen_seed
        if length is None:
            length = self.config.gen_length

        seed_ids = self.data.encode(seed)
        generated_ids = self.network.generate(seed_ids, length, temperature)
        return self.data.decode(generated_ids)
