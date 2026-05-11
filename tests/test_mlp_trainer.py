from atlas_ai.training.mlp_trainer import (
    MLPTrainer,
)


def test_mlp_training():

    trainer = MLPTrainer(
        learning_rate=0.05,
        epochs=20,
    )

    results = trainer.train()

    assert "final_loss" in results
    assert "loss_history" in results

    assert results["final_loss"] >= 0
