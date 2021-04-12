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
from collections import defaultdict
from astropy.coordinates import SkyCoord
import astropy.units as u
from zort.source import Source as zort_source

from puzle import app
from puzle import db
from puzle import login
from puzle.catalog import fetch_ogle_target


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
    db.Column('star_id', db.String(128), db.ForeignKey('puzle.star.id')),
    schema='puzle'
)


user_cand_association = db.Table(
    'user_candidate_association',
    db.Column('user_id', db.Integer, db.ForeignKey('puzle.user.id')),
    db.Column('candidate_id', db.String(128), db.ForeignKey('puzle.candidate.id')),
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
    candidates = db.relationship('Candidate', secondary=user_cand_association,
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
    
    def follow_candidate(self, cand):
        if not self.is_following_candidate(cand):
            self.candidates.append(cand)

    def unfollow_candidate(self, cand):
        if self.is_following_candidate(cand):
            self.candidates.remove(cand)

    def is_following_candidate(self, cand):
        return cand in self.candidates.all()

    def followed_candidates(self):
        return Candidate.query.join(user_cand_association,
            (user_cand_association.c.candidate_id == Candidate.id)).\
            filter(user_cand_association.c.user_id == self.id).\
            order_by(Candidate.id.asc())


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
    fit_filter = db.Column(db.String(128))
    fit_t_0 = db.Column(db.Float)
    fit_t_E = db.Column(db.Float)
    fit_f_0 = db.Column(db.Float)
    fit_f_1 = db.Column(db.Float)
    fit_a_type = db.Column(db.String(128))
    fit_chi_squared_flat = db.Column(db.Float)
    fit_chi_squared_delta = db.Column(db.Float)
    cand_id = db.Column(db.String(128))

    def __init__(self, object_id_g, object_id_r, object_id_i,
                 lightcurve_position_g, lightcurve_position_r, lightcurve_position_i,
                 ra, dec, lightcurve_filename, ingest_job_id,
                 version='DR5', id=None, comments=None, _ztf_ids=None,
                 fit_filter=None, fit_t_0=None,
                 fit_t_E=None, fit_f_0=None,
                 fit_f_1=None, fit_a_type=None,
                 fit_chi_squared_flat=None,
                 fit_chi_squared_delta=None, cand_id=None, lightcurve_file_pointer=None):
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
        self.version = version
        self.id = id
        self.comments = comments
        self.zort_source = self.load_zort_source(lightcurve_file_pointer)
        self._ztf_ids = _ztf_ids
        self.fit_filter = fit_filter
        self.fit_t_0 = fit_t_0
        self.fit_t_E = fit_t_E
        self.fit_f_0 = fit_f_0
        self.fit_f_1 = fit_f_1
        self.fit_a_type = fit_a_type
        self.fit_chi_squared_flat = fit_chi_squared_flat
        self.fit_chi_squared_delta = fit_chi_squared_delta
        self.cand_id = cand_id
        
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
    def glonlat(self):
        try:
            return self._glonlat
        except AttributeError:
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

    def load_zort_source(self, lightcurve_file_pointer=None):
        dir_path_puzle = os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))
        dir_path = f'{dir_path_puzle}/data/DR5'
        fname = '%s/%s' % (dir_path, os.path.basename(self.lightcurve_filename))
        source = zort_source(filename=fname,
                             lightcurve_position_g=self.lightcurve_position_g,
                             lightcurve_position_r=self.lightcurve_position_r,
                             lightcurve_position_i=self.lightcurve_position_i,
                             lightcurve_file_pointer=lightcurve_file_pointer,
                             check_initialization=False)
        return source

    def load_lightcurve_plot(self):
        job_id = self.id.split('_')[0]
        job_id_prefix = job_id[:3]
        folder = f'puzle/static/source/{job_id_prefix}/{job_id}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        lightcurve_plot_filename = f'{folder}/{self.id}_lightcurve.png'
        if not os.path.exists(lightcurve_plot_filename):
            if self.fit_t_0:
                model_params = {self.fit_filter: {'t_0': self.fit_t_0,
                                                  't_E': self.fit_t_E,
                                                  'a_type': self.fit_a_type,
                                                  'f_0': self.fit_f_0,
                                                  'f_1': self.fit_f_1}
                                }
            else:
                model_params = None
            self.zort_source.plot_lightcurves(filename=lightcurve_plot_filename,
                                              model_params=model_params)

    def _fetch_mars_results(self):
        radius_deg = 2. / 3600.
        cone = '%f,%f,%f' % (self.ra, self.dec, radius_deg)
        query = {"queries": [{"cone": cone}]}
        results = requests.post('https://mars.lco.global/', json=query).json()
        if results['total'] == 0:
            return None
        else:
            return results

    def fetch_ztf_ids(self):
        results = self._fetch_mars_results()
        if results is None:
            return 0
        ztf_ids = [str(r['objectId']) for r in
                   results['results'][0]['results']]
        ztf_ids = list(set(ztf_ids))
        self.ztf_ids = None
        for ztf_id in ztf_ids:
            self.ztf_ids = ztf_id

        return len(ztf_ids)

    def fetch_ztf_sgscore(self):
        results = self._fetch_mars_results()
        if results is None:
            return None

        sgscore = results['results'][0]['results'][0]['candidate']['sgscore1']
        return sgscore


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

    @hybrid_method
    def cone_search(self, ra, dec, radius=2):
        radius_deg = radius / 3600.
        return func.q3c_radial_query(text('ra_start'), text('dec_start'), ra, dec, radius_deg)


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


class StarProcessJob(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    priority = db.Column(db.Integer, nullable=True)
    started = db.Column(db.Boolean, nullable=False, server_default='f')
    finished = db.Column(db.Boolean, nullable=False, server_default='f')
    uploaded = db.Column(db.Boolean, nullable=False, server_default='f')
    datetime_started = db.Column(db.DateTime, nullable=True)
    datetime_finished = db.Column(db.DateTime, nullable=True)
    slurm_job_id = db.Column(db.Integer, nullable=True)
    slurm_job_rank = db.Column(db.Integer, nullable=True)
    source_ingest_job_id = db.Column(db.BigInteger, nullable=False)
    num_stars = db.Column(db.Integer, nullable=True)
    num_stars_pass_n_days = db.Column(db.Integer, nullable=True)
    num_objs = db.Column(db.Integer, nullable=True)
    num_objs_pass_n_days = db.Column(db.Integer, nullable=True)
    num_objs_pass_eta = db.Column(db.Integer, nullable=True)
    num_stars_pass_eta = db.Column(db.Integer, nullable=True)
    num_objs_pass_rf = db.Column(db.Integer, nullable=True)
    num_stars_pass_rf = db.Column(db.Integer, nullable=True)
    num_objs_pass_eta_residual = db.Column(db.Integer, nullable=True)
    num_stars_pass_eta_residual = db.Column(db.Integer, nullable=True)
    epoch_edges = db.Column(db.JSON, nullable=True)
    eta_thresholds_low = db.Column(db.JSON, nullable=True)
    eta_thresholds_high = db.Column(db.JSON, nullable=True)
    num_candidates = db.Column(db.Integer, nullable=True)

    def __init__(self, source_ingest_job_id):
        self.source_ingest_job_id = source_ingest_job_id


class Star(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.String(128), primary_key=True, nullable=False)
    source_ids = db.Column(db.ARRAY(db.String(128)))
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    ingest_job_id = db.Column(db.BigInteger, nullable=False)
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))

    def __init__(self, source_ids, ra, dec,
                 ingest_job_id=None, id=None,
                 comments=None, _ztf_ids=None):
        self.source_ids = source_ids
        self.ra = ra
        self.dec = dec
        self.ingest_job_id = ingest_job_id
        self.id = id
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
        try:
            return self._glonlat
        except AttributeError:
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


class Candidate(db.Model):
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.String(128), primary_key=True, nullable=False)
    source_id_arr = db.Column(db.ARRAY(db.String(128)))
    color_arr = db.Column(db.ARRAY(db.String(8)))
    pass_arr = db.Column(db.ARRAY(db.Boolean))
    eta_best = db.Column(db.Float)
    rf_score_best = db.Column(db.Float)
    eta_residual_best = db.Column(db.Float)
    eta_threshold_low_best = db.Column(db.Float)
    eta_threshold_high_best = db.Column(db.Float)
    t_E_best = db.Column(db.Float)
    t_0_best = db.Column(db.Float)
    f_0_best = db.Column(db.Float)
    f_1_best = db.Column(db.Float)
    a_type_best = db.Column(db.String(128))
    chi_squared_flat_best = db.Column(db.Float)
    chi_squared_delta_best = db.Column(db.Float)
    idx_best = db.Column(db.Integer)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    ingest_job_id = db.Column(db.BigInteger, nullable=False)
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))
    num_objs_pass = db.Column(db.Integer)
    num_objs_tot = db.Column(db.Integer)
    ogle_target = db.Column(db.String(128))

    def __init__(self, source_id_arr, ra, dec,
                 ingest_job_id, id,
                 color_arr, pass_arr, eta_best,
                 rf_score_best, eta_residual_best,
                 eta_threshold_low_best, eta_threshold_high_best,
                 t_E_best, t_0_best, f_0_best,
                 f_1_best, a_type_best,
                 chi_squared_flat_best, chi_squared_delta_best,
                 idx_best, num_objs_pass, num_objs_tot,
                 comments=None, _ztf_ids=None, ogle_target=None):
        self.source_id_arr = source_id_arr
        self.color_arr = color_arr
        self.pass_arr = pass_arr
        self.eta_best = eta_best
        self.rf_score_best = rf_score_best
        self.eta_residual_best = eta_residual_best
        self.eta_threshold_low_best = eta_threshold_low_best
        self.eta_threshold_high_best = eta_threshold_high_best
        self.t_E_best = t_E_best
        self.t_0_best = t_0_best
        self.f_0_best = f_0_best
        self.f_1_best = f_1_best
        self.a_type_best = a_type_best
        self.chi_squared_flat_best = chi_squared_flat_best
        self.chi_squared_delta_best = chi_squared_delta_best
        self.idx_best = idx_best
        self.num_objs_pass = num_objs_pass
        self.num_objs_tot = num_objs_tot
        self.ra = ra
        self.dec = dec
        self.ingest_job_id = ingest_job_id
        self.id = id
        self.comments = comments
        self._ztf_ids = _ztf_ids
        self._glonlat = None
        self.ogle_target = ogle_target

    def __repr__(self):
        str = 'Candidate \n'
        str += f'Ra/Dec: ({self.ra:.5f}, {self.dec:.5f}) \n'
        for i, source_id in enumerate(self.source_id_arr, 1):
            str += f'Source {i} ID: {source_id} \n'
        return str

    @hybrid_property
    def glonlat(self):
        try:
            return self._glonlat
        except AttributeError:
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

    def _fetch_mars_results(self):
        radius_deg = 2. / 3600.
        cone = '%f,%f,%f' % (self.ra, self.dec, radius_deg)
        query = {"queries": [{"cone": cone}]}
        results = requests.post('https://mars.lco.global/', json=query).json()
        if results['total'] == 0:
            return None
        else:
            return results

    def fetch_ztf_ids(self):
        results = self._fetch_mars_results()
        if results is None:
            return 0
        ztf_ids = [str(r['objectId']) for r in
                   results['results'][0]['results']]
        ztf_ids = list(set(ztf_ids))
        self.ztf_ids = None
        for ztf_id in ztf_ids:
            self.ztf_ids = ztf_id

        return len(ztf_ids)

    @hybrid_method
    def return_source_dct(self):
        source_dct = defaultdict(list)
        for source_id, color, pass_id, idx in zip(self.source_id_arr, self.color_arr, self.pass_arr, range(self.num_objs_tot)):
            source_dct[source_id].append((color, pass_id, idx))
        return source_dct

    @property
    def best_source_id(self):
        best_source_id = None
        for i, (source_id, color) in enumerate(zip(self.source_id_arr, self.color_arr)):
            if i == self.idx_best:
                best_source_id = source_id
        return best_source_id

    @property
    def unique_source_id_arr(self):
        return list(set(self.source_id_arr))

    def fetch_ogle_target(self):
        self.ogle_target = fetch_ogle_target(self.ra, self.dec)
        return self.ogle_target
