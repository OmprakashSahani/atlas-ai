import json
from pathlib import Path


CHECKPOINT_DIR = Path("checkpoints")


class CheckpointManager:
    """
    Save and load model parameters.
    """

    def __init__(self):

        CHECKPOINT_DIR.mkdir(
            exist_ok=True
        )

    def save_checkpoint(
        self,
        model,
        filename: str,
    ) -> str:

        parameters = [
            parameter.data
            for parameter in model.parameters()
        ]

        checkpoint_data = {
            "parameters": parameters
        }

        path = (
            CHECKPOINT_DIR
            / f"{filename}.json"
        )

        with open(path, "w") as f:
            json.dump(
                checkpoint_data,
                f,
                indent=4,
            )

        return str(path)

    def load_checkpoint(
        self,
        model,
        filename: str,
    ):

        path = (
            CHECKPOINT_DIR
            / f"{filename}.json"
        )

        with open(path, "r") as f:
            checkpoint_data = json.load(f)

        parameters = checkpoint_data[
            "parameters"
        ]

        for parameter, value in zip(
            model.parameters(),
            parameters,
        ):
            parameter.data = value

        return model
