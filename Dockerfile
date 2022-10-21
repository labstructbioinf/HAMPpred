FROM python:3.9.12-bullseye as base

WORKDIR /


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install Redis
COPY hamp_pred hamp_pred
COPY external external
CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "hamp_pred.app.run:app"]