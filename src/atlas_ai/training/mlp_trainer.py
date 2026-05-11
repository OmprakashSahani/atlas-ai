from atlas_ai.autograd.value import Value
from atlas_ai.checkpoint.checkpoint_manager import CheckpointManager
from atlas_ai.nn.modules import MLP
from atlas_ai.tracker.experiment_tracker import ExperimentTracker


class MLPTrainer:
    """
    Train a small MLP using autograd.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 50,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self):
        tracker = ExperimentTracker()

        run_id = tracker.create_run(
            name="mlp_training",
            config={
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
            },
        )

        model = MLP(3, [4, 4, 1])

        training_data = [
            ([2.0, 3.0, -1.0], 1.0),
            ([3.0, -1.0, 0.5], -1.0),
            ([0.5, 1.0, 1.0], -1.0),
            ([1.0, 1.0, -1.0], 1.0),
        ]

        loss_history = []

        for epoch in range(self.epochs):
            predictions = []

            for x, _ in training_data:
                inputs = [Value(v) for v in x]
                prediction = model(inputs)
                predictions.append(prediction)

            losses = []

            for prediction, (_, target) in zip(
                predictions,
                training_data,
            ):
                target_value = Value(target)
                loss = (prediction - target_value) ** 2
                losses.append(loss)

            total_loss = sum(
                losses,
                Value(0.0),
            )

            for parameter in model.parameters():
                parameter.grad = 0.0

            total_loss.backward()

            for parameter in model.parameters():
                parameter.data -= (
                    self.learning_rate
                    * parameter.grad
                )

            average_loss = (
                total_loss.data
                / len(training_data)
            )

            loss_history.append(average_loss)

        tracker.log_metric(
            run_id=run_id,
            metric_name="final_loss",
            value=loss_history[-1],
        )

        checkpoint_manager = CheckpointManager()

        checkpoint_path = checkpoint_manager.save_checkpoint(
            model,
            "trained_mlp",
        )

        return {
            "run_id": run_id,
            "final_loss": round(
                loss_history[-1],
                6,
            ),
            "checkpoint_path": checkpoint_path,
            "loss_history": loss_history,
        }