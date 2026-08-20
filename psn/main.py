"""Main entry point for the Phase Siren Network.

Usage:
    python -m psn.main              # Train on Shakespeare with defaults
    python -m psn.main --epochs 20  # Train for 20 epochs
    python -m psn.main --gen        # Generate text from a trained model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psn.config import PSNConfig
from psn.training.trainer import PSNTrainer
from psn.metrics.metrics import export_for_web_demo


def main():
    # Configuration optimized for demo / prototype
    config = PSNConfig(
        num_experts=8,
        kuramoto_K=4.0,
        dt=0.01,
        num_ticks=10,
        phase_dim=64,
        siren_hidden_dim=128,
        embedding_dim=32,
        hash_dim=32,
        routing_top_k=2,
        seq_len=128,
        batch_size=2,
        num_epochs=10,
        adaptation_lr=0.005,
        freq_adaptation=0.001,
        clone_threshold=0.95,
        prune_threshold=0.05,
        log_interval=5,
        gen_seed="HAMLET ",
        gen_length=300,
        save_metrics_path="download/psn_metrics.jsonl",
    )

    # Train
    trainer = PSNTrainer(config)
    metrics = trainer.train(num_epochs=config.num_epochs)

    # Export for web demo
    export_for_web_demo(
        metrics,
        "download/psn_demo_data.json",
    )

    # Generate final sample
    print("\n" + "=" * 60)
    print("Final Generation:")
    print("=" * 60)
    text = trainer.generate_text(temperature=0.7)
    print(text[:500])
    print("...")


if __name__ == "__main__":
    main()
