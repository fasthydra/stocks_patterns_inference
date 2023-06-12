FROM python:3.9

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install poetry

COPY poetry.lock pyproject.toml /code/
RUN POETRY_VIRTUALENVS_CREATE=false \
  && poetry install --no-interaction --no-ansi

COPY ./.env /code/
COPY ./inference.py /code/





