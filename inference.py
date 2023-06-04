import  os
from dotenv import load_dotenv

import pandas as pd
import numpy as np

import mlflow
from fastapi import FastAPI, File, UploadFile, HTTPException

load_dotenv()

app = FastAPI()

class Model:
    def __init__(self, model_name, model_stage):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction


model = Model("KShape", "Staging")

@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        # data = np.load(file.filename)

        csv_data = np.loadtxt(file.filename, delimiter=',')
        original_shape = (csv_data.shape[0], 30, 1)
        data = np.reshape(csv_data, original_shape)

        os.remove(file.filename)

        return model.predict(data[:10]).tolist()
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files accepted.")
