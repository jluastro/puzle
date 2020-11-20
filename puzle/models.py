from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.expression import func
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy import text
from flask_login import UserMixin
from time import time
import jwt
import os
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
from zort.object import Object as zort_object

from puzle import app
from puzle import db
from puzle import login


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
        return f"User('{self.username}', '{self.email}')"

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

    def __repr__(self):
        return f"Object(id: '{self.id}', nepochs: '{self.nepochs} \n " \
               f"lightcurve_filename: {self.lightcurve_filename} \n " \
               f"lightcurve_position: {self.lightcurve_position}')"


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
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))

    object_g = db.relationship('Object', foreign_keys=[object_id_g],
                               backref=db.backref('object_g', lazy='dynamic'))
    object_r = db.relationship('Object', foreign_keys=[object_id_r],
                               backref=db.backref('object_r', lazy='dynamic'))
    object_i = db.relationship('Object', foreign_keys=[object_id_i],
                               backref=db.backref('object_i', lazy='dynamic'))

    def __init__(self, object_id_g, object_id_r, object_id_i,
                 lightcurve_position_g, lightcurve_position_r,
                 lightcurve_position_i, ra, dec, lightcurve_filename,
                 comments=None, _ztf_ids=None):
        self.object_id_g = object_id_g
        self.object_id_r = object_id_r
        self.object_id_i = object_id_i
        self.lightcurve_position_g = lightcurve_position_g
        self.lightcurve_position_r = lightcurve_position_r
        self.lightcurve_position_i = lightcurve_position_i
        self.ra = ra
        self.dec = dec
        self.lightcurve_filename = lightcurve_filename
        self.comments = comments
        self._ztf_ids = _ztf_ids


    @hybrid_property
    def glon(self):
        coord = SkyCoord(self.ra, self.dec, unit=u.degree, frame='icrs')
        return coord.galactic.l.value

    @hybrid_property
    def glat(self):
        coord = SkyCoord(self.ra, self.dec, unit=u.degree, frame='icrs')
        return coord.galactic.b.value

    @property
    def ztf_ids(self):
        return [x for x in self._ztf_ids.split(';') if len(x) != 0]

    @ztf_ids.setter
    def ztf_ids(self, ztf_id):
        if ztf_id:
            self._ztf_ids += ';%s' % ztf_id
        else:
            self._ztf_ids = ''

    @hybrid_method
    def cone_search(self, ra, dec, radius=2/3600.):
        return func.q3c_radial_query(text('ra'), text('dec'), ra, dec, radius)

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

    def load_lightcurve_plot(self):
        folder = f'puzle/static/source/{self.id}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        lightcurve_plot_filename = f'{folder}/lightcurve.png'
        if not os.path.exists(lightcurve_plot_filename):
            self.set_parent()
            self.load_zort_object()
            self.zort_object.plot_lightcurves(filename=lightcurve_plot_filename)

    def fetch_ztf_ids(self):
        radius_deg = 2. / 3600.
        cone = '%f,%f,%f' % (self.ra, self.dec, radius_deg)
        query = {"queries": [{"cone": cone}]}
        results = requests.post('https://mars.lco.global/', json=query).json()
        if results['total'] == 0:
            return 0

        ztf_ids = [str(r['objectId']) for r in
                   results['results'][0]['results']]
        ztf_ids = list(set(ztf_ids))
        self.ztf_ids = None
        for ztf_id in ztf_ids:
            self.ztf_ids = ztf_id

        return len(ztf_ids)


class SourceIngestJob(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    lightcurve_filename = db.Column(db.String(128), index=True, nullable=False)
    process_rank = db.Column(db.Integer, nullable=False)
    process_size = db.Column(db.Integer, nullable=False)
    started = db.Column(db.Boolean, nullable=False, default=False)
    ended = db.Column(db.Boolean, nullable=False, default=False)
    datetime = db.Column(db.DateTime, nullable=True)
    slurm_job_id = db.Column(db.Integer, nullable=True)

    def __init__(self, lightcurve_filename, process_rank, process_size):
        self.lightcurve_filename = lightcurve_filename
        self.process_rank = process_rank
        self.process_size = process_size
