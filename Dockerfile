FROM python:3.9.12-bullseye as base

WORKDIR /


COPY pyproject.toml pyproject.toml
RUN pip install poetry
RUN poetry install --with=server

COPY hamp_pred hamp_pred
COPY external external
CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "hamp_pred.app.run:app"]