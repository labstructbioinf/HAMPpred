from flask_restx import Api
from celery import Celery
BROKER_URL = "redis://cache:6379"
celery = Celery(__name__, broker=BROKER_URL, backend=BROKER_URL)
api = Api(version='1.0', title='HAMP Pred',
          description='A predictor for hamp crick deviation',
          )
from hamp_pred.app.api.ml_models.endpoints.models import ns as models_endpoint
from hamp_pred.app.api.tasks.endpoints.status import ns as tasks_status_endpoint

api.add_namespace(models_endpoint)
api.add_namespace(tasks_status_endpoint)