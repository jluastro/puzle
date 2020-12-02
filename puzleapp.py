import logging
from logging.handlers import SMTPHandler, RotatingFileHandler
import datetime
import os

from puzle import app, db
from puzle.models import User


@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User}


app.logger.info('PUZLE Webapp startup')

if not app.debug:
    if app.config['MAIL_SERVER']:
        auth = None
        if app.config['MAIL_USERNAME'] or app.config['MAIL_PASSWORD']:
            auth = (
            app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        secure = None
        if app.config['MAIL_USE_TLS']:
            secure = ()
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        mail_handler = SMTPHandler(
            mailhost=(app.config['MAIL_SERVER'], app.config['MAIL_PORT']),
            fromaddr='no-reply@' + app.config['MAIL_SERVER'],
            toaddrs=app.config['ADMINS'],
            subject='PUZLE Webapp Failure: %s' % date,
            credentials=auth, secure=secure)
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)

if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/puzle_webapp.log', maxBytes=10240,
                                   backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)