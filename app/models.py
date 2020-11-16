from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from time import time
import jwt
from app import app
from app import db
from app import login


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class User(UserMixin, db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {'reset_password': self.id, 'exp': time() + expires_in},
            app.config['SECRET_KEY'], algorithm='HS256').decode('utf-8')

    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(token, app.config['SECRET_KEY'],
                            algorithms=['HS256'])['reset_password']
        except:
            return
        return User.query.get(id)


class Object(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=False)
    nepochs = db.Column(db.Integer, nullable=False)
    filterid = db.Column(db.Integer, index=True, nullable=False)
    fieldid = db.Column(db.Integer, nullable=False)
    rcid = db.Column(db.Integer, nullable=False)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    lightcurve_position = db.Column(db.BigInteger, nullable=False)
    lightcurve_filename = db.Column(db.String(128), index=True, nullable=False)


class Source(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    object_id_g = db.Column(db.BigInteger, db.ForeignKey('puzle.object.id'))
    object_id_r = db.Column(db.BigInteger, db.ForeignKey('puzle.object.id'))
    object_id_i = db.Column(db.BigInteger, db.ForeignKey('puzle.object.id'))
    lightcurve_position_g = db.Column(db.BigInteger)
    lightcurve_position_r = db.Column(db.BigInteger)
    lightcurve_position_i = db.Column(db.BigInteger)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    lightcurve_filename = db.Column(db.String(128), index=True, nullable=False)

    object_g = db.relationship('Object', foreign_keys=[object_id_g],
                               backref=db.backref('object_g', lazy='dynamic'))
    object_r = db.relationship('Object', foreign_keys=[object_id_r],
                               backref=db.backref('object_r', lazy='dynamic'))
    object_i = db.relationship('Object', foreign_keys=[object_id_i],
                               backref=db.backref('object_i', lazy='dynamic'))
