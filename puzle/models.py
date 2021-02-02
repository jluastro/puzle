from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.expression import func
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy import text, orm
from flask_login import UserMixin
from time import time
import jwt
import os
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
from zort.source import Source as zort_source

from puzle import app
from puzle import db
from puzle import login


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


user_source_association = db.Table(
    'user_source_association',
    db.Column('user_id', db.Integer, db.ForeignKey('puzle.user.id')),
    db.Column('source_id', db.String(128), db.ForeignKey('puzle.source.id')),
    schema='puzle'
)

user_star_association = db.Table(
    'user_star_association',
    db.Column('user_id', db.Integer, db.ForeignKey('puzle.user.id')),
    db.Column('star_id', db.BigInteger, db.ForeignKey('puzle.star.id')),
    schema='puzle'
)


class User(UserMixin, db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    sources = db.relationship('Source', secondary=user_source_association,
                              lazy='dynamic',
                              backref=db.backref('users', lazy='dynamic'))
    stars = db.relationship('Star', secondary=user_star_association,
                            lazy='dynamic',
                            backref=db.backref('users', lazy='dynamic'))

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

    def follow_source(self, source):
        if not self.is_following_source(source):
            self.sources.append(source)

    def unfollow_source(self, source):
        if self.is_following_source(source):
            self.sources.remove(source)

    def is_following_source(self, source):
        return source in self.sources.all()

    def followed_sources(self):
        return Source.query.join(user_source_association,
            (user_source_association.c.source_id == Source.id)).\
            filter(user_source_association.c.user_id == self.id).\
            order_by(Source.id.asc())
    
    def follow_star(self, star):
        if not self.is_following_star(star):
            self.stars.append(star)

    def unfollow_star(self, star):
        if self.is_following_star(star):
            self.stars.remove(star)

    def is_following_star(self, star):
        return star in self.stars.all()

    def followed_stars(self):
        return Star.query.join(user_star_association,
            (user_star_association.c.star_id == Star.id)).\
            filter(user_star_association.c.user_id == self.id).\
            order_by(Star.id.asc())


class Source(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.String(128), primary_key=True, nullable=False)
    object_id_g = db.Column(db.BigInteger)
    object_id_r = db.Column(db.BigInteger)
    object_id_i = db.Column(db.BigInteger)
    lightcurve_position_g = db.Column(db.BigInteger)
    lightcurve_position_r = db.Column(db.BigInteger)
    lightcurve_position_i = db.Column(db.BigInteger)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    ingest_job_id = db.Column(db.BigInteger, nullable=False)
    lightcurve_filename = db.Column(db.String(128), index=True, nullable=False)
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))

    def __init__(self, object_id_g, object_id_r, object_id_i,
                 lightcurve_position_g, lightcurve_position_r, lightcurve_position_i,
                 ra, dec, lightcurve_filename, ingest_job_id,
                 id=None, comments=None, _ztf_ids=None):
        self.object_id_g = object_id_g
        self.object_id_r = object_id_r
        self.object_id_i = object_id_i
        self.lightcurve_position_g = lightcurve_position_g
        self.lightcurve_position_r = lightcurve_position_r
        self.lightcurve_position_i = lightcurve_position_i
        self.ra = ra
        self.dec = dec
        self.lightcurve_filename = lightcurve_filename
        self.ingest_job_id = ingest_job_id
        self.id = id
        self.comments = comments
        self.zort_source = self.load_zort_source()
        self._ztf_ids = _ztf_ids
        
    def __repr__(self):
        return f'Source \n' \
               f'Ra/Dec: ({self.ra:.5f}, {self.dec:.5f})\n' \
               f'Filename: {self.lightcurve_filename} \n' \
               f'Object-g ID: {self.object_id_g} \n' \
               f'Object-r ID: {self.object_id_r} \n' \
               f'Object-i ID: {self.object_id_i} \n' \

    @orm.reconstructor
    def init_on_load(self):
        self.zort_source = self.load_zort_source()

    @hybrid_property
    def glon(self):
        coord = SkyCoord(self.ra, self.dec, unit=u.degree, frame='icrs')
        glon = coord.galactic.l.value
        if glon > 180:
            return glon - 360
        else:
            return glon

    @hybrid_property
    def glat(self):
        coord = SkyCoord(self.ra, self.dec, unit=u.degree, frame='icrs')
        glat = coord.galactic.b.value
        return glat

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
    def cone_search(self, ra, dec, radius=2):
        radius_deg = radius / 3600.
        return func.q3c_radial_query(text('ra'), text('dec'), ra, dec, radius_deg)

    def load_zort_source(self):
        dir_path_puzle = os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))
        dir_path_DR3 = f'{dir_path_puzle}/data/DR3'
        fname = '%s/%s' % (dir_path_DR3, os.path.basename(self.lightcurve_filename))
        source = zort_source(filename=fname,
                             lightcurve_position_g=self.lightcurve_position_g,
                             lightcurve_position_r=self.lightcurve_position_r,
                             lightcurve_position_i=self.lightcurve_position_i)
        return source

    def load_lightcurve_plot(self):
        folder = f'puzle/static/source/{self.id}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        lightcurve_plot_filename = f'{folder}/lightcurve.png'
        if not os.path.exists(lightcurve_plot_filename):
            self.zort_source.plot_lightcurves(filename=lightcurve_plot_filename)

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
    ra_start = db.Column(db.Float, nullable=False)
    ra_end = db.Column(db.Float, nullable=False)
    dec_start = db.Column(db.Float, nullable=False)
    dec_end = db.Column(db.Float, nullable=False)
    started = db.Column(db.Boolean, nullable=False, server_default='f')
    finished = db.Column(db.Boolean, nullable=False, server_default='f')
    uploaded = db.Column(db.Boolean, nullable=False, server_default='f')
    datetime_started = db.Column(db.DateTime, nullable=True)
    datetime_finished = db.Column(db.DateTime, nullable=True)
    slurm_job_id = db.Column(db.Integer, nullable=True)
    slurm_job_rank = db.Column(db.Integer, nullable=True)

    def __init__(self, ra_start, ra_end, dec_start, dec_end):
        self.ra_start = ra_start
        self.ra_end = ra_end
        self.dec_start = dec_start
        self.dec_end = dec_end


class StarIngestJob(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    started = db.Column(db.Boolean, nullable=False, server_default='f')
    finished = db.Column(db.Boolean, nullable=False, server_default='f')
    uploaded = db.Column(db.Boolean, nullable=False, server_default='f')
    datetime_started = db.Column(db.DateTime, nullable=True)
    datetime_finished = db.Column(db.DateTime, nullable=True)
    slurm_job_id = db.Column(db.Integer, nullable=True)
    slurm_job_rank = db.Column(db.Integer, nullable=True)
    source_ingest_job_id = db.Column(db.BigInteger, nullable=False)

    def __init__(self, source_ingest_job_id):
        self.source_ingest_job_id = source_ingest_job_id


class Star(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    source_ids = db.Column(db.ARRAY(db.String(128)))
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    ingest_job_id = db.Column(db.BigInteger, nullable=False)
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))

    def __init__(self, source_ids, ra, dec,
                 ingest_job_id=None,
                 comments=None, _ztf_ids=None):
        self.source_ids = source_ids
        self.ra = ra
        self.dec = dec
        self.ingest_job_id = ingest_job_id
        self.comments = comments
        self._ztf_ids = _ztf_ids
        self._glonlat = None

    def __repr__(self):
        str = 'Star \n'
        str += f'Ra/Dec: ({self.ra:.5f}, {self.dec:.5f}) \n'
        for i, source_id in enumerate(self.source_ids, 1):
            str += f'Source {i} ID: {source_id} \n'
        return str

    @hybrid_property
    def glonlat(self):
        if self._glonlat is None:
            coord = SkyCoord(self.ra, self.dec, unit=u.degree, frame='icrs')
            glon, glat = coord.galactic.l.value, coord.galactic.b.value
            if glon > 180:
                glon -= 360
            self._glonlat = (glon, glat)
        return self._glonlat

    @property
    def glon(self):
        return self.glonlat[0]

    @property
    def glat(self):
        return self.glonlat[1]

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
    def cone_search(self, ra, dec, radius=2):
        radius_deg = radius / 3600.
        return func.q3c_radial_query(text('ra'), text('dec'), ra, dec, radius_deg)
