from flask import Blueprint, Flask

from hamp_pred.app.celery_config import make_celery

app = Flask(__name__)
app.config.update(CELERY_CONFIG={
    'broker_url': 'redis://cache:6379',
    'result_backend': 'redis://cache:6379',
})


def init_app(app):
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)
    app.register_blueprint(blueprint)


celery = make_celery(app)
from hamp_pred.app.api.rest import api

init_app(app)
