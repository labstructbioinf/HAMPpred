from flask_restx import Resource

from hamp_pred.app.api.ml_models.models.models import model, model_input_sequences
from hamp_pred.app.api.rest import api
from hamp_pred.app.async_tasks.model import predict_async
from hamp_pred.src.predictor import Predictor

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
        results = predict_async.apply_async(args=[model_name, seqs])
        return results.id
