from celery.result import AsyncResult
from flask_restx import Resource

from hamp_pred.app.api.rest import api

ns = api.namespace('tasks')


@ns.route('/status/<string:task_id>')
class TaskStatus(Resource):
    @ns.doc('Check results')
    def get(self, task_id):
        res = AsyncResult(task_id)
        if res.ready():
            return res.get()
        else:
            return "progress"
