from pathlib import Path

import matplotlib.pyplot as plt

from atlas_ai.training.mlp_trainer import (
    MLPTrainer,
)


OUTPUT_DIR = Path("benchmarks")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    trainer = MLPTrainer(
        learning_rate=0.05,
        epochs=50,
    )

    results = trainer.train()

    loss_history = results["loss_history"]

    epochs = list(
        range(1, len(loss_history) + 1)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        loss_history,
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "MLP Training Loss Convergence"
    )

    output_path = (
        OUTPUT_DIR / "mlp_loss_curve.png"
    )

    plt.savefig(output_path)

    print(
        f"Loss curve saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
