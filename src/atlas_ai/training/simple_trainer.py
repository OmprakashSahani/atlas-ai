import random
import time

from atlas_ai.tracker.experiment_tracker import (
    ExperimentTracker,
)


class SimpleLinearModel:
    """
    Simple linear model: y = wx + b
    """

    def __init__(self):
        self.weight = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)

    def predict(self, x: float) -> float:
        return self.weight * x + self.bias


class SimpleTrainer:
    """
    Train a simple linear regression model
    using gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def mean_squared_error(
        self,
        prediction: float,
        target: float,
    ) -> float:
        return (prediction - target) ** 2

    def train(self):
        """
        Train on synthetic data:
        y = 2x + 1
        """

        tracker = ExperimentTracker()

        run_id = tracker.create_run(
            name="linear_regression_training",
            config={
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
            },
        )

        start_time = time.perf_counter()

        model = SimpleLinearModel()

        training_data = [
            (1, 3),
            (2, 5),
            (3, 7),
            (4, 9),
            (5, 11),
        ]

        losses = []

        for epoch in range(self.epochs):

            total_loss = 0

            for x, y in training_data:

                prediction = model.predict(x)

                loss = self.mean_squared_error(
                    prediction,
                    y,
                )

                total_loss += loss

                grad_prediction = 2 * (
                    prediction - y
                )

                grad_weight = grad_prediction * x
                grad_bias = grad_prediction

                model.weight -= (
                    self.learning_rate * grad_weight
                )

                model.bias -= (
                    self.learning_rate * grad_bias
                )

            average_loss = (
                total_loss / len(training_data)
            )

            losses.append(average_loss)

        end_time = time.perf_counter()

        training_time = end_time - start_time

        tracker.log_metric(
            run_id=run_id,
            metric_name="final_loss",
            value=losses[-1],
        )

        tracker.log_metric(
            run_id=run_id,
            metric_name="training_time_sec",
            value=training_time,
        )

        return {
            "run_id": run_id,
            "final_weight": round(model.weight, 4),
            "final_bias": round(model.bias, 4),
            "final_loss": round(losses[-1], 6),
            "training_time_sec": round(
                training_time,
                6,
            ),
            "loss_history": losses,
        }
