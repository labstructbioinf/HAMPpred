from flask import Blueprint, Flask
from hamp_pred.app.api.rest import api

app = Flask(__name__)


def init_app(app):
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)
    app.register_blueprint(blueprint)


init_app(app)

