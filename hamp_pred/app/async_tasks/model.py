import json

from hamp_pred.app.app import celery
from hamp_pred.src.predictor import Predictor
from hamp_pred.utils.numpy_json import NumpyEncoder


@celery.task(bind=True)
def predict_async(celer, model_name, sequences):
    """Async long task method."""
    result = Predictor(model_name).predict(sequences)
    result = json.loads(json.dumps(result.to_dict(orient='records'), cls=NumpyEncoder))
    return result