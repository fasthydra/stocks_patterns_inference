FROM python:3.9

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install poetry

COPY poetry.lock pyproject.toml /code/
RUN POETRY_VIRTUALENVS_CREATE=false \
  && poetry install --no-interaction --no-ansi

COPY ./.env /code/
COPY ./src /code/src
COPY ./inference.py /code/

CMD ["poetry", "run", "uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8001"]


