import json
import time

from celery.result import AsyncResult
from flask_restx import Resource

from hamp_pred.app.api.ml_models.models.models import model, model_input_sequences
from hamp_pred.app.api.rest import api, celery
from hamp_pred.src.predictor import Predictor
from hamp_pred.utils.numpy_json import NumpyEncoder

ns = api.namespace('models')


@ns.route('/')
class ModelsInfo(Resource):
    @ns.doc('list all models')
    @ns.marshal_list_with(model)
    def get(self):
        models = Predictor.get_models_info()
        return models


@ns.route('/<string:model_name>/predict')
class RunPrediction(Resource):
    @ns.doc('Run prediction on sequence')
    @ns.expect(model_input_sequences, validate=True)
    def post(self, model_name):
        seqs = ns.payload['sequences']
        results = self.predict_async.apply_async(args=[model_name, seqs])
        return results.id

    @staticmethod
    @celery.task(bind=True)
    def predict_async(celer, model_name, sequences):
        """Async long task method."""
        result = Predictor(model_name).predict(sequences)
        result = json.loads(json.dumps(result.to_dict(orient='records'), cls=NumpyEncoder))
        return result
