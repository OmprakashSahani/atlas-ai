from fastapi import FastAPI

from atlas_ai.autograd.value import Value
from atlas_ai.checkpoint.checkpoint_manager import (
    CheckpointManager,
)
from atlas_ai.monitoring.inference_metrics import (
    InferenceMetrics,
)
from atlas_ai.nn.modules import MLP


app = FastAPI(
    title="Atlas AI Inference Server"
)

model = MLP(3, [4, 4, 1])

metrics = InferenceMetrics()

checkpoint_manager = (
    CheckpointManager()
)

try:

    checkpoint_manager.load_checkpoint(
        model,
        "trained_mlp",
    )

    checkpoint_status = (
        "Loaded trained checkpoint"
    )

except FileNotFoundError:

    checkpoint_status = (
        "No trained checkpoint found"
    )


@app.get("/")
def root():

    return {
        "message": (
            "Atlas AI Inference Server"
        ),
        "checkpoint_status": (
            checkpoint_status
        ),
    }


@app.post("/predict")
def predict(features: list[float]):

    start_time = (
        metrics.start_timer()
    )

    inputs = [
        Value(v)
        for v in features
    ]

    prediction = model(inputs)

    latency = metrics.stop_timer(
        start_time
    )

    return {
        "prediction": prediction.data,
        "latency_sec": round(
            latency,
            6,
        ),
    }


@app.get("/metrics")
def get_metrics():

    return metrics.summary()
