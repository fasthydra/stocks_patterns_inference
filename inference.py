import base64
import os

import mlflow
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_NAME = "KShape"
MODEL_STAGE = "Staging"
EXPERIMENT_NAME = "mlflow_test"

load_dotenv()

app = FastAPI()

server_url = os.environ.get("MLFLOW_TRACKING_URI")
username = os.environ.get("NGINX_LOGIN")
password = os.environ.get("NGINX_PASSWORD")

# Кодирование учетных данных в base64
basic_auth = base64.b64encode(f"{username}:{password}".encode()).decode()

# Формирование полного URI, включая аутентификационные данные
auth_server_url = f"http://{username}:{password}@{server_url}"

# Установка адреса сервера с учетными данными
mlflow.set_tracking_uri(auth_server_url)


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
