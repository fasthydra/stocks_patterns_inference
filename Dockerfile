FROM python:3.9

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install poetry

COPY poetry.lock pyproject.toml /code/
COPY ./src /code/
COPY ./inference.py /code/
COPY ./.env /code/

RUN POETRY_VIRTUALENVS_CREATE=false \
  && poetry install --no-interaction --no-ansi


CMD ["uviconr", "inference:app", "--host", "http://127.0.0.1", "--port", "80"]
