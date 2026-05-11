from fastapi import FastAPI

from atlas_ai.autograd.value import Value
from atlas_ai.nn.modules import MLP


app = FastAPI(
    title="Atlas AI Inference Server"
)

model = MLP(3, [4, 4, 1])


@app.get("/")
def root():

    return {
        "message": "Atlas AI Inference Server"
    }


@app.post("/predict")
def predict(features: list[float]):

    inputs = [
        Value(v)
        for v in features
    ]

    prediction = model(inputs)

    return {
        "prediction": prediction.data
    }
