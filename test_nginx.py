import base64
import os

import mlflow
from dotenv import load_dotenv

load_dotenv()

server_url = os.environ.get("MLFLOW_TRACKING_URI")
username = os.environ.get("NGINX_LOGIN")
password = os.environ.get("NGINX_PASSWORD")

MODEL_NAME = "KShape"
MODEL_STAGE = "Staging"

# Кодирование учетных данных в base64
basic_auth = base64.b64encode(f"{username}:{password}".encode()).decode()

# Формирование полного URI, включая аутентификационные данные
auth_server_url = f"http://{username}:{password}@{server_url}"

# Установка адреса сервера с учетными данными
mlflow.set_tracking_uri(auth_server_url)

# Загрузка модели
model_name = MODEL_NAME
model_stage = MODEL_STAGE
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_stage}"
)

print(model)
