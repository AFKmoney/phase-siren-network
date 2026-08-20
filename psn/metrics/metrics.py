"""Metrics for the Phase Siren Network.

The PSN has a rich set of metrics beyond standard accuracy and loss:

1. **Kuramoto Order Parameter (r)**: Measures global phase synchronization.
   r near 0 = incoherent (good diversity), r near 1 = synchronized.

2. **Routing Entropy**: Measures how evenly tokens are distributed across experts.
   High entropy = good load balancing.

3. **Phase Coherence Spectrum**: FFT of expert phases over time.
   Shows dominant frequencies and synchronization patterns.

4. **Expert Utilization Gini**: Measures inequality in expert utilization.
   High Gini = some experts are overworked.

5. **Bits Per Character (BPC)**: Information-theoretic measure of prediction quality.
   Lower = better. Random = log2(vocab_size).

6. **Phase Velocity**: How fast expert phases are changing.
   High velocity = active dynamics, low = frozen state.

7. **Synchronization Rate**: How quickly inter-instance sync occurs.

8. **Self-Modification Rate**: How many adaptations per step.

These metrics are exported as JSONL for the web demo to visualize.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List
import math


def compute_order_parameter(phases: jax.Array) -> float:
    """Compute Kuramoto order parameter r.

    Args:
        phases: Phase array, shape (N,).

    Returns:
        r in [0, 1].
    """
    z = jnp.mean(jnp.exp(1j * phases))
    return float(jnp.abs(z))


def compute_routing_entropy(routing_weights: jax.Array) -> float:
    """Compute entropy of routing distribution.

    Args:
        routing_weights: (batch, num_experts) routing weights.

    Returns:
        Shannon entropy.
    """
    avg_routing = jnp.mean(routing_weights, axis=0)
    avg_routing = avg_routing / (jnp.sum(avg_routing) + 1e-8)
    return float(-jnp.sum(avg_routing * jnp.log(avg_routing + 1e-8)))


def compute_gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient for expert utilization inequality.

    Args:
        values: List of utilization values.

    Returns:
        Gini coefficient in [0, 1]. 0 = perfect equality, 1 = perfect inequality.
    """
    if len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total < 1e-8:
        return 0.0
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / total) / n
    return float(max(0, gini))


def compute_phase_velocity(phases_history: List[List[float]]) -> float:
    """Compute mean phase velocity (rad/tick).

    Args:
        phases_history: List of phase snapshots, each a list of floats.

    Returns:
        Mean angular velocity in rad/tick.
    """
    if len(phases_history) < 2:
        return 0.0
    velocities = []
    for t in range(1, len(phases_history)):
        N = min(len(phases_history[t]), len(phases_history[t - 1]))
        for i in range(N):
            diff = phases_history[t][i] - phases_history[t - 1][i]
            # Handle wrap-around
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            velocities.append(abs(diff))
    return float(np.mean(velocities)) if velocities else 0.0


def compute_bits_per_char(logits: jax.Array, targets: jax.Array) -> float:
    """Compute bits per character.

    BPC = -mean(log2(softmax(logits)[targets]))

    Args:
        logits: (batch, seq_len, vocab_size) or (batch, vocab_size).
        targets: (batch, seq_len) or (batch,).

    Returns:
        Bits per character (lower = better, random = log2(vocab_size)).
    """
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits)
    nll = -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])
    return float(nll / jnp.log(2.0))


def compute_all_metrics(metrics_log: List[Dict[str, Any]], step_range: tuple = None
) -> Dict[str, Any]:
    """Compute summary statistics from the full metrics log.

    Args:
        metrics_log: List of per-step metric dictionaries.
        step_range: Optional (start, end) to filter steps.

    Returns:
        Dictionary with comprehensive summary statistics.
    """
    if step_range:
        log = metrics_log[step_range[0]:step_range[1]]
    else:
        log = metrics_log

    if not log:
        return {}

    accuracies = [m["accuracy"] for m in log if "accuracy" in m]
    bpcs = [m["bits_per_char"] for m in log if "bits_per_char" in m]
    order_params = [m["order_parameter"] for m in log if "order_parameter" in m]
    routing_entropies = [m.get("routing_entropy", 0) for m in log]
    active_experts = [m.get("active_experts", 0) for m in log]
    modifications = [m.get("modifications", 0) for m in log]

    result = {
        "num_steps": len(log),
        "final_accuracy": accuracies[-1] if accuracies else 0.0,
        "best_accuracy": max(accuracies) if accuracies else 0.0,
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "final_bpc": bpcs[-1] if bpcs else 0.0,
        "best_bpc": min(bpcs) if bpcs else 0.0,
        "mean_bpc": float(np.mean(bpcs)) if bpcs else 0.0,
        "final_order_param": order_params[-1] if order_params else 0.0,
        "mean_order_param": float(np.mean(order_params)) if order_params else 0.0,
        "mean_routing_entropy": float(np.mean(routing_entropies)) if routing_entropies else 0.0,
        "final_active_experts": active_experts[-1] if active_experts else 0,
        "total_modifications": sum(modifications),
        "accuracy_history": accuracies,
        "bpc_history": bpcs,
        "order_param_history": order_params,
    }

    # Compute Gini for expert utilization at final step
    last_step = log[-1]
    if "expert_utilizations" in last_step and len(last_step["expert_utilizations"]) > 1:
        result["utilization_gini"] = compute_gini_coefficient(
            last_step["expert_utilizations"]
        )

    return result


def export_for_web_demo(metrics_log: List[Dict[str, Any]], output_path: str):
    """Export metrics in a format suitable for the web demo.

    Creates a JSON file with time-series data for all metrics.

    Args:
        metrics_log: List of per-step metric dictionaries.
        output_path: Path to save the JSON file.
    """
    import json
    import os

    summary = compute_all_metrics(metrics_log)

    # Time series data for charts
    time_series = {
        "steps": [],
        "accuracy": [],
        "bpc": [],
        "order_parameter": [],
        "active_experts": [],
        "routing_entropy": [],
        "expert_phases": [],
    }

    for m in metrics_log:
        time_series["steps"].append(m.get("step", 0))
        time_series["accuracy"].append(m.get("accuracy", 0))
        time_series["bpc"].append(m.get("bits_per_char", 0))
        time_series["order_parameter"].append(m.get("order_parameter", 0))
        time_series["active_experts"].append(m.get("active_experts", 0))
        time_series["routing_entropy"].append(m.get("routing_entropy", 0))
        # Expert phases for the phase wheel visualization
        if "expert_phases" in m:
            time_series["expert_phases"].append(m["expert_phases"])

    export_data = {
        "summary": summary,
        "time_series": time_series,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"[Metrics] Exported demo data to {output_path}")
