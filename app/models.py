from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from time import time
import jwt
import os
from zort.object import Object as zort_object
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

    def set_parent(self):
        object_ids = [self.object_id_g, self.object_id_r, self.object_id_i]
        lightcurve_positions = [self.lightcurve_position_g,
                                self.lightcurve_position_r,
                                self.lightcurve_position_i]
        self.parent_id = None
        self.parent_lightcurve_position = None
        self.child_ids = []
        self.child_lightcurve_positions = []
        for i in range(3):
            if object_ids[i] is None:
                continue

            self.parent_id = object_ids[i]
            self.parent_lightcurve_position = lightcurve_positions[i]

            for j in range(i+1, 3):
                if object_ids[j] is None:
                    continue

                self.child_ids.append(object_ids[j])
                self.child_lightcurve_positions.append(lightcurve_positions[j])

            break

    def load_zort_object(self):
        dir_path_puzle = os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))
        dir_path_DR3 = f'{dir_path_puzle}/data/DR3'
        fname = '%s/%s' % (dir_path_DR3, self.lightcurve_filename)
        obj = zort_object(fname, self.parent_lightcurve_position)
        for lc_pos in self.child_lightcurve_positions:
            sib = zort_object(fname, lc_pos)
            obj.siblings.append(sib)
        self.zort_object = obj

    def set_lightcurve_plot_filename(self):
        folder = f'app/static/sources/{self.id}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        lightcurve_plot_filename = f'{folder}/lightcurve.png'
        if not os.path.exists(lightcurve_plot_filename):
            self.set_parent()
            self.load_zort_object()
            self.zort_object.plot_lightcurves(filename=lightcurve_plot_filename)
        self.lightcurve_plot_filename = \
            lightcurve_plot_filename.replace('app', '')
