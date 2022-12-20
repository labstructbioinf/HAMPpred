from flask_restx import fields

from hamp_pred.app.api.rest import api

model = api.model('Model', {
    'name': fields.String(readonly=True, description='Model name'),
})

model_input_sequences = api.model('Sequence', {
    'sequences': fields.List(fields.String(), required=True),
})
