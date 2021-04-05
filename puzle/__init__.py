import os
from flask import Flask
from config import Config
from sqlalchemy.pool import NullPool
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_mail import Mail
from flask_bootstrap import Bootstrap

os.environ['PATH'] += os.pathsep + '/global/common/shared/das/container_proxy'

app = Flask(__name__)
app.config.from_object(Config)

app.static_url_path = app.config.get('STATIC_FOLDER')
app.static_folder = app.root_path + app.static_url_path

db = SQLAlchemy(app, engine_options={'poolclass': NullPool})
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'
mail = Mail(app)
bootstrap = Bootstrap(app)

from puzle import routes, models, errors, ulensdb
