import os

import requests

url = "http://127.0.0.1:8001/invocations"
file_path = "data/sber_clst_2021_1_1.30_10.csv"

with open(os.path.abspath(file_path), "rb") as file:
    files = {"file": (file_path, file, "application/octet-stream")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    prediction = response.content.decode("utf-8")
    print("Prediction:", prediction)
else:
    print("Error:", response.text)
