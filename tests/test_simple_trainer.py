from atlas_ai.training.simple_trainer import (
    SimpleTrainer,
)


def test_simple_training():
    trainer = SimpleTrainer(
        learning_rate=0.01,
        epochs=100,
    )

    results = trainer.train()

    assert "final_weight" in results
    assert "final_bias" in results
    assert "final_loss" in results

    assert results["final_loss"] < 1.0
