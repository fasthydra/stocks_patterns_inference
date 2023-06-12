import http.client
import os

HOST = "localhost"
PORT = 8000
METHOD = "POST"
URL = "/invocations"
TEST_FILE = "test_inference.csv"

file_path = os.path.abspath(TEST_FILE)

# Открываем файл в бинарном режиме и считываем его содержимое
with open(file_path, "rb") as file:
    file_content = file.read()

# Создаем соединение HTTP
conn = http.client.HTTPConnection(HOST, PORT)

# Определяем заголовки запроса
headers = {
    "Content-Type": "multipart/form-data; boundary=boundary",
}

# Формируем тело запроса с правильным форматом multipart/form-data
body = (
    b'--boundary\r\nContent-Disposition: form-data; name="file";'
    b'filename="file.csv"\r\nContent-Type: application/octet-stream\r\n\r\n'
)
body += file_content
body += b"\r\n--boundary--"

# Отправляем запрос
conn.request(METHOD, URL, body=body, headers=headers)

# Получаем ответ
response = conn.getresponse()
# Читаем и декодируем содержимое ответа
response_content = response.read().decode()

if response.status == 200:
    # prediction = response.content.decode("utf-8")
    print("Prediction:", response_content)
else:
    print("Error:", response.text)

# Закрываем соединение
conn.close()
