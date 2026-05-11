from atlas_ai.checkpoint.checkpoint_manager import (
    CheckpointManager,
)
from atlas_ai.nn.modules import MLP


def test_checkpoint_save_and_load():

    manager = CheckpointManager()

    model = MLP(3, [4, 4, 1])

    original_parameters = [
        p.data
        for p in model.parameters()
    ]

    manager.save_checkpoint(
        model,
        "test_model",
    )

    for parameter in model.parameters():
        parameter.data = 0.0

    manager.load_checkpoint(
        model,
        "test_model",
    )

    restored_parameters = [
        p.data
        for p in model.parameters()
    ]

    assert (
        original_parameters
        == restored_parameters
    )
