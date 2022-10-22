from flask_restx import Api

api = Api(version='1.0', title='HAMP Pred',
          description='A predictor for hamp crick deviation',
          )
from hamp_pred.app.api.ml_models.endpoints.models import ns as models_endpoint
from hamp_pred.app.api.tasks.endpoints.status import ns as tasks_status_endpoint

api.add_namespace(models_endpoint)
api.add_namespace(tasks_status_endpoint)