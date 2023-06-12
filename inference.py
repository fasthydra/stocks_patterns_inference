import os

import mlflow
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_NAME = "KShape"
MODEL_STAGE = "Staging"
EXPERIMENT_NAME = "test_dvc_new"

load_dotenv()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

app = FastAPI()


class Model:
    def __init__(self, model_name, model_stage):
        self.model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{model_stage}"
        )

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction


model = Model(MODEL_NAME, MODEL_STAGE)


@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())

        csv_data = np.loadtxt(file.filename, delimiter=",")
        original_shape = (csv_data.shape[0], 30, 1)
        data = np.reshape(csv_data, original_shape)

        os.remove(file.filename)

        return model.predict(data).tolist()
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV files accepted.",
        )


@app.get("/")
def read_root():
    return {f"{mlflow.set_experiment(EXPERIMENT_NAME)}": "name"}