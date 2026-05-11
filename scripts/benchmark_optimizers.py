from pathlib import Path

import matplotlib.pyplot as plt

from atlas_ai.autograd.value import Value
from atlas_ai.nn.modules import MLP
from atlas_ai.optim.optimizers import (
    Momentum,
    SGD,
)


OUTPUT_DIR = Path("benchmarks")
OUTPUT_DIR.mkdir(exist_ok=True)


TRAINING_DATA = [
    ([2.0, 3.0, -1.0], 1.0),
    ([3.0, -1.0, 0.5], -1.0),
    ([0.5, 1.0, 1.0], -1.0),
    ([1.0, 1.0, -1.0], 1.0),
]


def train_model(
    optimizer_class,
    optimizer_name,
):

    model = MLP(3, [4, 4, 1])

    optimizer = optimizer_class(
        model.parameters(),
        learning_rate=0.05,
    )

    loss_history = []

    epochs = 50

    for _ in range(epochs):

        predictions = []

        for x, _ in TRAINING_DATA:

            inputs = [
                Value(v)
                for v in x
            ]

            prediction = model(inputs)

            predictions.append(prediction)

        losses = []

        for prediction, (_, target) in zip(
            predictions,
            TRAINING_DATA,
        ):

            target_value = Value(target)

            loss = (
                prediction
                - target_value
            ) ** 2

            losses.append(loss)

        total_loss = sum(
            losses,
            Value(0.0),
        )

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        average_loss = (
            total_loss.data
            / len(TRAINING_DATA)
        )

        loss_history.append(
            average_loss
        )

    print(
        f"{optimizer_name} "
        f"final_loss="
        f"{loss_history[-1]:.6f}"
    )

    return loss_history


def main():

    sgd_losses = train_model(
        SGD,
        "SGD",
    )

    momentum_losses = train_model(
        Momentum,
        "Momentum",
    )

    epochs = list(
        range(
            1,
            len(sgd_losses) + 1,
        )
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        sgd_losses,
        label="SGD",
        marker="o",
    )

    plt.plot(
        epochs,
        momentum_losses,
        label="Momentum",
        marker="s",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Optimizer Convergence Comparison"
    )

    plt.legend()

    output_path = (
        OUTPUT_DIR
        / "optimizer_comparison.png"
    )

    plt.savefig(output_path)

    print(
        f"\nPlot saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
