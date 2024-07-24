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
import numpy as np
from collections import defaultdict
from zort.source import Source as zort_source
from zort.photometry import fluxes_to_magnitudes, magnitudes_to_fluxes

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
    db.Column('candidate_id', db.String(128), db.ForeignKey('puzle.candidate_level2.id')),
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
    candidates = db.relationship('CandidateLevel2', secondary=user_cand_association,
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
        if not self.is_following_candidate(cand.id):
            self.candidates.append(cand)

    def unfollow_candidate(self, cand):
        if self.is_following_candidate(cand.id):
            self.candidates.remove(cand)

    def is_following_candidate(self, candid):
        return candid in [c.id for c in self.candidates.all()]

    def followed_candidates(self):
        return CandidateLevel2.query.join(user_cand_association,
            (user_cand_association.c.candidate_id == CandidateLevel2.id)).\
            filter(user_cand_association.c.user_id == self.id).\
            order_by(CandidateLevel2.id.asc())


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
        from astropy.coordinates import SkyCoord
        import astropy.units as u
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

    def load_lightcurve_plot(self, folder=None, model_params=None, model=None):
        if folder is None:
            job_id = self.id.split('_')[0]
            job_id_prefix = job_id[:3]
            folder = f'puzle/static/source/{job_id_prefix}/{job_id}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        lightcurve_plot_filename = f'{folder}/{self.id}_lightcurve.png'
        if not os.path.exists(lightcurve_plot_filename):
            if model_params is None and self.fit_t_0 is not None:
                model_params = {self.fit_filter: {'t_0': self.fit_t_0,
                                                  't_E': self.fit_t_E,
                                                  'a_type': self.fit_a_type,
                                                  'f_0': self.fit_f_0,
                                                  'f_1': self.fit_f_1}
                                }
            self.zort_source.plot_lightcurves(filename=lightcurve_plot_filename,
                                              model_params=model_params,
                                              model=model,
                                              hmjd_survey_bounds=True)

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
        from astropy.coordinates import SkyCoord
        import astropy.units as u
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

class CandidateLevel2(db.Model):
    __tablename__ = 'candidate_level2'
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
        str = f'Candidate {self.id}\n'
        str += f'Ra/Dec: ({self.ra:.5f}, {self.dec:.5f}) \n'
        for i, source_id in enumerate(self.source_id_arr, 1):
            str += f'Source {i} ID: {source_id} \n'
        return str

    @hybrid_property
    def glonlat(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
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


class CandidateLevel3(db.Model):
    __tablename__ = 'candidate_level3'
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.String(128), primary_key=True, nullable=False)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    source_id_arr = db.Column(db.ARRAY(db.String(128)))
    color_arr = db.Column(db.ARRAY(db.String(8)))
    pass_arr = db.Column(db.ARRAY(db.Boolean))
    idx_best = db.Column(db.Integer)
    num_objs_pass = db.Column(db.Integer)
    num_objs_tot = db.Column(db.Integer)
    num_epochs_best = db.Column(db.Integer)
    num_days_best = db.Column(db.Integer)
    eta_best = db.Column(db.Float)
    eta_residual_best = db.Column(db.Float)
    t0_best = db.Column(db.Float)
    u0_amp_best = db.Column(db.Float)
    tE_best = db.Column(db.Float)
    mag_src_best = db.Column(db.Float)
    b_sff_best = db.Column(db.Float)
    piE_E_best = db.Column(db.Float)
    piE_N_best = db.Column(db.Float)
    chi_squared_ulens_best = db.Column(db.Float)
    chi_squared_flat_inside_1tE_best = db.Column(db.Float)
    chi_squared_flat_inside_2tE_best = db.Column(db.Float)
    chi_squared_flat_inside_3tE_best = db.Column(db.Float)
    chi_squared_flat_outside_1tE_best = db.Column(db.Float)
    chi_squared_flat_outside_2tE_best = db.Column(db.Float)
    chi_squared_flat_outside_3tE_best = db.Column(db.Float)
    num_days_inside_1tE_best = db.Column(db.Integer)
    num_days_inside_2tE_best = db.Column(db.Integer)
    num_days_inside_3tE_best = db.Column(db.Integer)
    num_days_outside_1tE_best = db.Column(db.Integer)
    num_days_outside_2tE_best = db.Column(db.Integer)
    num_days_outside_3tE_best = db.Column(db.Integer)
    delta_hmjd_outside_1tE_best = db.Column(db.Float)
    delta_hmjd_outside_2tE_best = db.Column(db.Float)
    delta_hmjd_outside_3tE_best = db.Column(db.Float)
    num_3sigma_peaks_inside_2tE_best = db.Column(db.Integer)
    num_5sigma_peaks_inside_2tE_best = db.Column(db.Integer)
    num_3sigma_peaks_outside_2tE_best = db.Column(db.Integer)
    num_5sigma_peaks_outside_2tE_best = db.Column(db.Integer)
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))
    ogle_target = db.Column(db.String(128))

    def __init__(self, id, ra, dec,
                 source_id_arr, color_arr,
                 pass_arr, idx_best,
                 num_objs_pass, num_objs_tot,
                 num_epochs_best, num_days_best,
                 eta_best=None, eta_residual_best=None,
                 t0_best=None, u0_amp_best=None,
                 tE_best=None, mag_src_best=None,
                 b_sff_best=None, piE_E_best=None, piE_N_best=None,
                 chi_squared_ulens_best=None,
                 chi_squared_flat_inside_1tE_best=None,
                 chi_squared_flat_inside_2tE_best=None,
                 chi_squared_flat_inside_3tE_best=None,
                 chi_squared_flat_outside_1tE_best=None,
                 chi_squared_flat_outside_2tE_best=None,
                 chi_squared_flat_outside_3tE_best=None,
                 num_days_inside_1tE_best=None,
                 num_days_inside_2tE_best=None,
                 num_days_inside_3tE_best=None,
                 num_days_outside_1tE_best=None,
                 num_days_outside_2tE_best=None,
                 num_days_outside_3tE_best=None,
                 delta_hmjd_outside_1tE_best=None,
                 delta_hmjd_outside_2tE_best=None,
                 delta_hmjd_outside_3tE_best=None,
                 num_3sigma_peaks_inside_2tE_best=None,
                 num_5sigma_peaks_inside_2tE_best=None,
                 num_3sigma_peaks_outside_2tE_best=None,
                 num_5sigma_peaks_outside_2tE_best=None,
                 comments=None, _ztf_ids=None, ogle_target=None):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.source_id_arr = source_id_arr
        self.color_arr = color_arr
        self.pass_arr = pass_arr
        self.idx_best = idx_best
        self.num_objs_pass = num_objs_pass
        self.num_objs_tot = num_objs_tot
        self.num_epochs_best = num_epochs_best
        self.num_days_best = num_days_best
        self.eta_best = eta_best
        self.eta_residual_best = eta_residual_best
        self.t0_best = t0_best
        self.u0_amp_best = u0_amp_best
        self.tE_best = tE_best
        self.mag_src_best = mag_src_best
        self.b_sff_best = b_sff_best
        self.piE_E_best = piE_E_best
        self.piE_N_best = piE_N_best
        self.chi_squared_ulens_best = chi_squared_ulens_best
        self.chi_squared_flat_inside_1tE_best = chi_squared_flat_inside_1tE_best
        self.chi_squared_flat_inside_2tE_best = chi_squared_flat_inside_2tE_best
        self.chi_squared_flat_inside_3tE_best = chi_squared_flat_inside_3tE_best
        self.chi_squared_flat_outside_1tE_best = chi_squared_flat_outside_1tE_best
        self.chi_squared_flat_outside_2tE_best = chi_squared_flat_outside_2tE_best
        self.chi_squared_flat_outside_3tE_best = chi_squared_flat_outside_3tE_best
        self.num_days_inside_1tE_best = num_days_inside_1tE_best
        self.num_days_inside_2tE_best = num_days_inside_2tE_best
        self.num_days_inside_3tE_best = num_days_inside_3tE_best
        self.num_days_outside_1tE_best = num_days_outside_1tE_best
        self.num_days_outside_2tE_best = num_days_outside_2tE_best
        self.num_days_outside_3tE_best = num_days_outside_3tE_best
        self.delta_hmjd_outside_1tE_best = delta_hmjd_outside_1tE_best
        self.delta_hmjd_outside_2tE_best = delta_hmjd_outside_2tE_best
        self.delta_hmjd_outside_3tE_best = delta_hmjd_outside_3tE_best
        self.num_3sigma_peaks_inside_2tE_best = num_3sigma_peaks_inside_2tE_best
        self.num_5sigma_peaks_inside_2tE_best = num_5sigma_peaks_inside_2tE_best
        self.num_3sigma_peaks_outside_2tE_best = num_3sigma_peaks_outside_2tE_best
        self.num_5sigma_peaks_outside_2tE_best = num_5sigma_peaks_outside_2tE_best
        self.comments = comments
        self._ztf_ids = _ztf_ids
        self.ogle_target = ogle_target

    def __repr__(self):
        str = f'Candidate {self.id}\n'
        str += f'Ra/Dec: ({self.ra:.5f}, {self.dec:.5f}) \n'
        return str

    @property
    def piE_best(self):
        if self.piE_E_best is None:
            return None
        else:
            return np.hypot(self.piE_E_best, self.piE_N_best)

    @hybrid_property
    def glonlat(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
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


class CandidateLevel4(db.Model):
    __tablename__ = 'candidate_level4'
    __table_args__ = {'schema': 'puzle'}

    id = db.Column(db.String(128), primary_key=True, nullable=False)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    source_id_arr = db.Column(db.ARRAY(db.String(128)))
    color_arr = db.Column(db.ARRAY(db.String(8)))
    pass_arr = db.Column(db.ARRAY(db.Boolean))
    idx_best = db.Column(db.Integer)
    num_objs_pass = db.Column(db.Integer)
    num_objs_tot = db.Column(db.Integer)
    num_epochs_arr = db.Column(db.ARRAY(db.Integer))
    num_days_arr = db.Column(db.ARRAY(db.Integer))
    eta_arr = db.Column(db.ARRAY(db.Float))
    eta_residual_arr = db.Column(db.ARRAY(db.Float))
    t0_arr = db.Column(db.ARRAY(db.Float))
    u0_amp_arr = db.Column(db.ARRAY(db.Float))
    tE_arr = db.Column(db.ARRAY(db.Float))
    mag_src_arr = db.Column(db.ARRAY(db.Float))
    b_sff_arr = db.Column(db.ARRAY(db.Float))
    piE_E_arr = db.Column(db.ARRAY(db.Float))
    piE_N_arr = db.Column(db.ARRAY(db.Float))
    chi_squared_ulens_arr = db.Column(db.ARRAY(db.Float))
    chi_squared_flat_arr = db.Column(db.ARRAY(db.Float))
    chi_squared_flat_inside_arr = db.Column(db.ARRAY(db.Float))
    chi_squared_flat_outside_arr = db.Column(db.ARRAY(db.Float))
    num_days_inside_arr = db.Column(db.ARRAY(db.Integer))
    num_days_outside_arr = db.Column(db.ARRAY(db.Integer))
    delta_hmjd_outside_arr = db.Column(db.ARRAY(db.Float))
    num_3sigma_peaks_inside_arr = db.Column(db.ARRAY(db.Integer))
    num_3sigma_peaks_outside_arr = db.Column(db.ARRAY(db.Integer))
    num_5sigma_peaks_inside_arr = db.Column(db.ARRAY(db.Integer))
    num_5sigma_peaks_outside_arr = db.Column(db.ARRAY(db.Integer))
    pspl_gp_fit_started = db.Column(db.Boolean, nullable=False, server_default='f')
    pspl_gp_fit_finished = db.Column(db.Boolean, nullable=False, server_default='f')
    pspl_gp_fit_datetime_started = db.Column(db.DateTime, nullable=True)
    pspl_gp_fit_datetime_finished = db.Column(db.DateTime, nullable=True)
    slurm_job_id = db.Column(db.Integer, nullable=True)
    node = db.Column(db.String(64))
    num_pspl_gp_fit_lightcurves = db.Column(db.Integer)
    fit_type_pspl_gp = db.Column(db.String(128))
    source_id_arr_pspl_gp = db.Column(db.ARRAY(db.String(128)))
    color_arr_pspl_gp = db.Column(db.ARRAY(db.String(8)))
    chi2_pspl_gp = db.Column(db.Float)
    rchi2_pspl_gp = db.Column(db.Float)
    logL_pspl_gp = db.Column(db.Float)
    t0_pspl_gp = db.Column(db.Float)
    t0_err_pspl_gp = db.Column(db.Float)
    u0_amp_pspl_gp = db.Column(db.Float)
    u0_amp_err_pspl_gp = db.Column(db.Float)
    tE_pspl_gp = db.Column(db.Float)
    tE_err_pspl_gp = db.Column(db.Float)
    piE_E_pspl_gp = db.Column(db.Float)
    piE_E_err_pspl_gp = db.Column(db.Float)
    piE_N_pspl_gp = db.Column(db.Float)
    piE_N_err_pspl_gp = db.Column(db.Float)
    piE_pspl_gp = db.Column(db.Float)
    piE_err_pspl_gp = db.Column(db.Float)
    b_sff_pspl_gp = db.Column(db.Float)
    b_sff_err_pspl_gp = db.Column(db.Float)
    mag_base_pspl_gp = db.Column(db.Float)
    mag_base_err_pspl_gp = db.Column(db.Float)
    gp_log_sigma_pspl_gp = db.Column(db.Float)
    gp_log_sigma_err_pspl_gp = db.Column(db.Float)
    gp_rho_pspl_gp = db.Column(db.Float)
    gp_rho_err_pspl_gp = db.Column(db.Float)
    gp_log_omega04_S0_pspl_gp = db.Column(db.Float)
    gp_log_omega04_S0_err_pspl_gp = db.Column(db.Float)
    gp_log_omega0_pspl_gp = db.Column(db.Float)
    gp_log_omega0_err_pspl_gp = db.Column(db.Float)
    b_sff_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    b_sff_err_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    mag_base_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    mag_base_err_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_log_sigma_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_log_sigma_err_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_rho_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_rho_err_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_log_omega04_S0_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_log_omega04_S0_err_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_log_omega0_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    gp_log_omega0_err_arr_pspl_gp = db.Column(db.ARRAY(db.Float))
    category = db.Column(db.String(128))
    level5 = db.Column(db.Boolean, nullable=False, server_default='f')
    ongoing = db.Column(db.Boolean, nullable=False, server_default='f')
    delta_hmjd_outside_pspl_gp = db.Column(db.Float)
    comments = db.Column(db.String(1024))
    _ztf_ids = db.Column(db.String(256))
    ogle_target = db.Column(db.String(128))

    def __init__(self, id=None, ra=None, dec=None, source_id_arr=None, color_arr=None,
                 pass_arr=None, idx_best=None, num_objs_pass=None, num_objs_tot=None,
                 num_epochs_arr=None, num_days_arr=None, eta_arr=None,
                 eta_residual_arr=None, t0_arr=None, u0_amp_arr=None,
                 tE_arr=None, mag_src_arr=None, b_sff_arr=None, piE_E_arr=None,
                 piE_N_arr=None, chi_squared_ulens_arr=None, chi_squared_flat_arr=None,
                 chi_squared_flat_inside_arr=None,
                 chi_squared_flat_outside_arr=None,
                 num_days_inside_arr=None,
                 num_days_outside_arr=None,
                 delta_hmjd_outside_arr=None,
                 num_3sigma_peaks_inside_arr=None, num_3sigma_peaks_outside_arr=None,
                 num_5sigma_peaks_inside_arr=None, num_5sigma_peaks_outside_arr=None,
                 pspl_gp_fit_started=None, pspl_gp_fit_finished=None,
                 pspl_gp_fit_datetime_started=None, pspl_gp_fit_datetime_finished=None,
                 slurm_job_id=None, node=None, num_pspl_gp_fit_lightcurves=None,
                 source_id_arr_pspl_gp=None, color_arr_pspl_gp=None,
                 chi2_pspl_gp=None, rchi2_pspl_gp=None, logL_pspl_gp=None,
                 fit_type_pspl_gp=None, t0_pspl_gp=None, t0_err_pspl_gp=None,
                 u0_amp_pspl_gp=None, u0_amp_err_pspl_gp=None, tE_pspl_gp=None,
                 tE_err_pspl_gp=None, piE_E_pspl_gp=None, piE_E_err_pspl_gp=None,
                 piE_N_pspl_gp=None, piE_N_err_pspl_gp=None, piE_pspl_gp=None,
                 piE_err_pspl_gp=None, b_sff_pspl_gp=None,
                 b_sff_err_pspl_gp=None, mag_base_pspl_gp=None, mag_base_err_pspl_gp=None,
                 gp_log_sigma_pspl_gp=None, gp_log_sigma_err_pspl_gp=None, gp_rho_pspl_gp=None,
                 gp_rho_err_pspl_gp=None, gp_log_omega04_S0_pspl_gp=None, gp_log_omega04_S0_err_pspl_gp=None,
                 gp_log_omega0_pspl_gp=None, gp_log_omega0_err_pspl_gp=None, b_sff_arr_pspl_gp=None,
                 b_sff_err_arr_pspl_gp=None, mag_base_arr_pspl_gp=None, mag_base_err_arr_pspl_gp=None,
                 gp_log_sigma_arr_pspl_gp=None, gp_log_sigma_err_arr_pspl_gp=None, gp_rho_arr_pspl_gp=None,
                 gp_rho_err_arr_pspl_gp=None, gp_log_omega04_S0_arr_pspl_gp=None, gp_log_omega04_S0_err_arr_pspl_gp=None,
                 gp_log_omega0_arr_pspl_gp=None, gp_log_omega0_err_arr_pspl_gp=None,
                 delta_hmjd_outside_pspl_gp=None,
                 category=None, level5=None, ongoing=None,
                 comments=None, _ztf_ids=None, ogle_target=None):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.source_id_arr = source_id_arr
        self.color_arr = color_arr
        self.pass_arr = pass_arr
        self.idx_best = idx_best
        self.num_objs_pass = num_objs_pass
        self.num_objs_tot = num_objs_tot
        self.num_epochs_arr = num_epochs_arr
        self.num_days_arr = num_days_arr
        self.eta_arr = eta_arr
        self.eta_residual_arr = eta_residual_arr
        self.t0_arr = t0_arr
        self.u0_amp_arr = u0_amp_arr
        self.tE_arr = tE_arr
        self.mag_src_arr = mag_src_arr
        self.b_sff_arr = b_sff_arr
        self.piE_E_arr = piE_E_arr
        self.piE_N_arr = piE_N_arr
        self.chi_squared_ulens_arr = chi_squared_ulens_arr
        self.chi_squared_flat_arr = chi_squared_flat_arr
        self.chi_squared_flat_inside_arr = chi_squared_flat_inside_arr
        self.chi_squared_flat_outside_arr = chi_squared_flat_outside_arr
        self.num_days_inside_arr = num_days_inside_arr
        self.num_days_outside_arr = num_days_outside_arr
        self.delta_hmjd_outside_arr = delta_hmjd_outside_arr
        self.num_3sigma_peaks_inside_arr = num_3sigma_peaks_inside_arr
        self.num_3sigma_peaks_outside_arr = num_3sigma_peaks_outside_arr
        self.num_5sigma_peaks_inside_arr = num_5sigma_peaks_inside_arr
        self.num_5sigma_peaks_outside_arr = num_5sigma_peaks_outside_arr
        self.pspl_gp_fit_started = pspl_gp_fit_started
        self.pspl_gp_fit_finished = pspl_gp_fit_finished
        self.pspl_gp_fit_datetime_started = pspl_gp_fit_datetime_started
        self.pspl_gp_fit_datetime_finished = pspl_gp_fit_datetime_finished
        self.slurm_job_id = slurm_job_id
        self.node = node
        self.num_pspl_gp_fit_lightcurves = num_pspl_gp_fit_lightcurves
        self.fit_type_pspl_gp = fit_type_pspl_gp
        self.source_id_arr_pspl_gp = source_id_arr_pspl_gp
        self.color_arr_pspl_gp = color_arr_pspl_gp
        self.chi2_pspl_gp = chi2_pspl_gp
        self.rchi2_pspl_gp = rchi2_pspl_gp
        self.logL_pspl_gp = logL_pspl_gp
        self.t0_pspl_gp = t0_pspl_gp
        self.t0_err_pspl_gp = t0_err_pspl_gp
        self.u0_amp_pspl_gp = u0_amp_pspl_gp
        self.u0_amp_err_pspl_gp = u0_amp_err_pspl_gp
        self.tE_pspl_gp = tE_pspl_gp
        self.tE_err_pspl_gp = tE_err_pspl_gp
        self.piE_E_pspl_gp = piE_E_pspl_gp
        self.piE_E_err_pspl_gp = piE_E_err_pspl_gp
        self.piE_N_pspl_gp = piE_N_pspl_gp
        self.piE_N_err_pspl_gp = piE_N_err_pspl_gp
        self.piE_pspl_gp = piE_pspl_gp
        self.piE_err_pspl_gp = piE_err_pspl_gp
        self.b_sff_pspl_gp = b_sff_pspl_gp
        self.b_sff_err_pspl_gp = b_sff_err_pspl_gp
        self.mag_base_pspl_gp = mag_base_pspl_gp
        self.mag_base_err_pspl_gp = mag_base_err_pspl_gp
        self.gp_log_sigma_pspl_gp = gp_log_sigma_pspl_gp
        self.gp_log_sigma_err_pspl_gp = gp_log_sigma_err_pspl_gp
        self.gp_rho_pspl_gp = gp_rho_pspl_gp
        self.gp_rho_err_pspl_gp = gp_rho_err_pspl_gp
        self.gp_log_omega04_S0_pspl_gp = gp_log_omega04_S0_pspl_gp
        self.gp_log_omega04_S0_err_pspl_gp = gp_log_omega04_S0_err_pspl_gp
        self.gp_log_omega0_pspl_gp = gp_log_omega0_pspl_gp
        self.gp_log_omega0_err_pspl_gp = gp_log_omega0_err_pspl_gp
        self.b_sff_arr_pspl_gp = b_sff_arr_pspl_gp
        self.b_sff_err_arr_pspl_gp = b_sff_err_arr_pspl_gp
        self.mag_base_arr_pspl_gp = mag_base_arr_pspl_gp
        self.mag_base_err_arr_pspl_gp = mag_base_err_arr_pspl_gp
        self.gp_log_sigma_arr_pspl_gp = gp_log_sigma_arr_pspl_gp
        self.gp_log_sigma_err_arr_pspl_gp = gp_log_sigma_err_arr_pspl_gp
        self.gp_rho_arr_pspl_gp = gp_rho_arr_pspl_gp
        self.gp_rho_err_arr_pspl_gp = gp_rho_err_arr_pspl_gp
        self.gp_log_omega04_S0_arr_pspl_gp = gp_log_omega04_S0_arr_pspl_gp
        self.gp_log_omega04_S0_err_arr_pspl_gp = gp_log_omega04_S0_err_arr_pspl_gp
        self.gp_log_omega0_arr_pspl_gp = gp_log_omega0_arr_pspl_gp
        self.gp_log_omega0_err_arr_pspl_gp = gp_log_omega0_err_arr_pspl_gp
        self.delta_hmjd_outside_pspl_gp = delta_hmjd_outside_pspl_gp
        self.category = category
        self.level5 = level5
        self.ongoing = ongoing
        self.comments = comments
        self._ztf_ids = _ztf_ids
        self.ogle_target = ogle_target

    def __repr__(self):
        str = f'Candidate {self.id}\n'
        str += f'Ra/Dec: ({self.ra:.5f}, {self.dec:.5f}) \n'
        return str

    @hybrid_property
    def glonlat(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
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

    @property
    def pspl_gp_fit_dct(self):
        if self.source_id_arr_pspl_gp is None:
            return None
        pspl_gp_fit_dct = {}
        for idx in range(len(self.source_id_arr_pspl_gp)):
            source_id = self.source_id_arr_pspl_gp[idx]
            if source_id not in pspl_gp_fit_dct:
                pspl_gp_fit_dct[source_id] = {}
            color = self.color_arr_pspl_gp[idx]
            b_sff = self.b_sff_arr_pspl_gp[idx]
            mag_base = self.mag_base_arr_pspl_gp[idx]
            flux_base, _ = magnitudes_to_fluxes(mag_base)
            flux_src = flux_base * b_sff
            mag_src, _ = fluxes_to_magnitudes(flux_src)
            pspl_gp_fit_dct[source_id][color] = {'t0': self.t0_pspl_gp,
                                                 'u0_amp': self.u0_amp_pspl_gp,
                                                 'tE': self.tE_pspl_gp,
                                                 'piE_E': self.piE_E_pspl_gp,
                                                 'piE_N': self.piE_N_pspl_gp,
                                                 'b_sff': b_sff,
                                                 'mag_src': mag_src,
                                                 'raL': self.ra,
                                                 'decL': self.dec}
        return pspl_gp_fit_dct

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
