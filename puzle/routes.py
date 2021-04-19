from flask import render_template, flash, redirect, url_for, request, session
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from datetime import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u
from puzle import app, db
from puzle.forms import LoginForm, RegistrationForm, \
    EditProfileForm, ResetPasswordRequestForm, ResetPasswordForm, \
    EditCommentForm, RadialSearchForm, EmptyForm, \
    FilterSearchForm
from puzle.models import User, Source, Candidate
from puzle.email import send_password_reset_email


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()


@app.route('/')
@app.route('/home')
@login_required
def home():
    return render_template('home.html', title='Home')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('home')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/user/<username>', methods=['GET', 'POST'])
@login_required
def user(username):
    form = EmptyForm()
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    cands = user.followed_candidates().paginate(page, app.config['ITEMS_PER_PAGE'], False)
    next_url = url_for('user', username=username, page=cands.next_num) \
        if cands.has_next else None
    prev_url = url_for('user', username=username, page=cands.prev_num) \
        if cands.has_prev else None
    return render_template('user.html', user=user, form=form, cands=cands,
                           next_url=next_url, prev_url=prev_url)


@app.route('/users', methods=['GET'])
@login_required
def users():
    users = User.query.all()
    return render_template('users.html', users=users)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.', 'success')
        return redirect(url_for('user', username=current_user.username))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html',
                           title='Edit Profile',
                           form=form)


@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_password_reset_email(user)
        flash('Check your email for the instructions to reset your password',
              'success')
        return redirect(url_for('login'))
    return render_template('reset_password_request.html',
                           title='Reset Password', form=form)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_password_token(token)
    if not user:
        return redirect(url_for('home'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been reset.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)


@app.route('/source/<sourceid>')
@login_required
def source(sourceid):
    title = 'Source %s' % sourceid
    form = EmptyForm()
    source = Source.query.filter_by(id=sourceid).first_or_404()
    source.load_lightcurve_plot()
    return render_template('source.html', source=source,
                           form=form, title=title)


@app.route('/candidate/<candid>')
@login_required
def candidate(candid):
    title = 'Candidate %s' % candid
    form = EmptyForm()
    cand = Candidate.query.filter_by(id=candid).first_or_404()
    sources = []
    pass_dct = {}
    data = zip(cand.source_id_arr, cand.pass_arr, cand.color_arr)
    for source_id, passFlag, color in data:
        if source_id not in pass_dct:
            source = Source.query.filter(Source.id==source_id).first_or_404()
            source.load_lightcurve_plot()
            sources.append(source)
            pass_dct[source_id] = {}
        pass_dct[source_id][color] = passFlag
    return render_template('candidate.html', cand=cand, sources=sources,
                           pass_dct=pass_dct, form=form, title=title, zip=zip)


@app.route('/edit_source_comments/<sourceid>', methods=['GET', 'POST'])
@login_required
def edit_source_comments(sourceid):
    source = Source.query.filter_by(id=sourceid).first_or_404()
    form = EditCommentForm()
    if form.validate_on_submit():
        source.comments = form.comments.data
        db.session.commit()
        flash('Your changes have been saved.', 'success')
        return redirect(url_for('source', sourceid=sourceid))
    elif request.method == 'GET':
        form.comments.data = source.comments
    return render_template('edit_source_comments.html',
                           form=form)


@app.route('/edit_candidate_comments/<candid>', methods=['GET', 'POST'])
@login_required
def edit_candidate_comments(candid):
    cand = Candidate.query.filter_by(id=candid).first_or_404()
    form = EditCommentForm()
    if form.validate_on_submit():
        cand.comments = form.comments.data
        db.session.commit()
        flash('Your changes have been saved.', 'success')
        return redirect(url_for('candidate', candid=candid))
    elif request.method == 'GET':
        form.comments.data = cand.comments
    return render_template('edit_candidate_comments.html',
                           form=form)


@app.route('/fetch_source_ztf_ids/<sourceid>', methods=['POST'])
@login_required
def fetch_source_ztf_ids(sourceid):
    source = Source.query.filter_by(id=sourceid).first_or_404()
    n_ids = source.fetch_ztf_ids()
    flash('%i ZTF IDs Found' % n_ids, 'success')
    db.session.commit()
    return redirect(url_for('source', sourceid=sourceid))


@app.route('/fetch_candidate_ztf_ids/<candid>', methods=['POST'])
@login_required
def fetch_candidate_ztf_ids(candid):
    cand = Candidate.query.filter_by(id=candid).first_or_404()
    n_ids = cand.fetch_ztf_ids()
    flash('%i ZTF IDs Found' % n_ids, 'success')
    db.session.commit()
    return redirect(url_for('candidate', candid=candid))


@app.route('/fetch_candidate_ogle_target/<candid>', methods=['POST'])
@login_required
def fetch_candidate_ogle_target(candid):
    cand = Candidate.query.filter_by(id=candid).first_or_404()
    ogle_target = cand.fetch_ogle_target()
    if ogle_target:
        flash('OGLE Target Found', 'success')
    else:
        flash('No OGLE Target Found', 'success')
    db.session.commit()
    return redirect(url_for('candidate', candid=candid))


@app.route('/follow_source/<sourceid>', methods=['POST'])
@login_required
def follow_source(sourceid):
    form = EmptyForm()
    if form.validate_on_submit():
        source = Source.query.filter_by(id=sourceid).first()
        if user is None:
            flash('Source {} not found.'.format(source.id), 'danger')
            return redirect(url_for('source', sourceid=sourceid))
        current_user.follow_source(source)
        db.session.commit()
        flash('You are following Source {}'.format(source.id), 'success')
        return redirect(url_for('source', sourceid=sourceid))
    else:
        return redirect(url_for('source', sourceid=sourceid))


@app.route('/unfollow_source/<sourceid>', methods=['POST'])
@login_required
def unfollow_source(sourceid):
    form = EmptyForm()
    if form.validate_on_submit():
        source = Source.query.filter_by(id=sourceid).first()
        if user is None:
            flash('Source {} not found.'.format(source.id), 'danger')
            return redirect(url_for('source', sourceid=sourceid))
        current_user.unfollow_source(source)
        db.session.commit()
        flash('You are not following Source {}'.format(source.id), 'success')
        return redirect(url_for('source', sourceid=sourceid))
    else:
        return redirect(url_for('source', sourceid=sourceid))


@app.route('/follow_candidate/<candid>_<whichpage>', methods=['POST'])
@login_required
def follow_candidate(candid, whichpage):
    form = EmptyForm()
    if form.validate_on_submit():
        cand = Candidate.query.filter_by(id=candid).first()
        if user is None:
            flash('Candidate {} not found.'.format(cand.id), 'danger')
            return redirect(url_for('candidate', candid=candid))
        current_user.follow_candidate(cand)
        db.session.commit()
        flash('You are following Candidate {}'.format(cand.id), 'success')
        if whichpage == "same":
            url = '%s#%s' % (request.referrer, candid)
            return redirect(url)
        elif whichpage == "cand":
            return redirect(url_for('candidate', candid=candid))
    else:
        return redirect(url_for('candidate', candid=candid))


@app.route('/unfollow_candidate/<candid>_<whichpage>', methods=['POST'])
@login_required
def unfollow_candidate(candid, whichpage):
    form = EmptyForm()
    if form.validate_on_submit():
        cand = Candidate.query.filter_by(id=candid).first()
        if user is None:
            flash('Source {} not found.'.format(cand.id), 'danger')
            return redirect(url_for('candidate', candidateid=candid))
        current_user.unfollow_candidate(cand)
        db.session.commit()
        flash('You are not following Source {}'.format(cand.id), 'success')
        if whichpage == "same":
            url = '%s#%s' % (request.referrer, candid)
            return redirect(url)
        elif whichpage == "cand":
            return redirect(url_for('candidate', candid=candid))
    else:
        return redirect(url_for('candidate', candid=candid))


@app.route('/sources', methods=['GET', 'POST'])
@login_required
def sources():
    page = request.args.get('page', 1, type=int)
    sources = Source.query.order_by(Source.id.asc()).paginate(page, app.config['ITEMS_PER_PAGE'], False)
    next_url = url_for('sources', page=sources.next_num) \
        if sources.has_next else None
    prev_url = url_for('sources', page=sources.prev_num) \
        if sources.has_prev else None
    return render_template('sources.html', sources=sources,
                           next_url=next_url, prev_url=prev_url,
                           title='PUZLE sources')


@app.route('/candidates', methods=['GET', 'POST'])
@login_required
def candidates():
    form = EmptyForm()
    page = request.args.get('page', 1, type=int)
    cands = Candidate.query.order_by(Candidate.eta_best.asc()).\
        paginate(page, app.config['ITEMS_PER_PAGE'], False)
    next_url = url_for('candidates', page=cands.next_num) \
        if cands.has_next else None
    prev_url = url_for('candidates', page=cands.prev_num) \
        if cands.has_prev else None

    for cand in cands.items:
        sources = Source.query.filter(Source.id.in_(cand.source_id_arr)).all()
        for source in sources:
            source.load_lightcurve_plot()

    return render_template('candidates.html', cands=cands,
                           next_url=next_url, prev_url=prev_url,
                           title='PUZLE candidates', zip=zip,
                           paginate=True, form=form)


@app.route('/radial_search', methods=['GET', 'POST'])
@login_required
def radial_search():
    form_radial = RadialSearchForm()
    glon, glat = None, None
    if form_radial.validate_on_submit():
        radius = form_radial.radius.data

        if form_radial.ra.data and form_radial.dec.data:
            ra, dec = form_radial.ra.data, form_radial.dec.data
            flash('Results for (ra, dec, radius) = (%.5f, %.5f, %.2f)' % (ra, dec, radius),
                  'info')
        elif form_radial.glon.data and form_radial.glat.data:
            glon, glat = form_radial.glon.data, form_radial.glat.data
            flash('Results for (glon, glat, radius) = (%.5f, %.5f, %.2f)' % (glon, glat, radius),
                  'info')
            coord = SkyCoord(glon, glat,
                             unit=u.degree, frame='galactic')
            ra = coord.icrs.ra.value
            dec = coord.icrs.dec.value
        else:
            flash('Either (ra, dec) or '
                  '(glon, glat) '
                  'must be entered.', 'danger')
            return redirect(url_for('radial_search'))

        session['radius'] = radius
        session['ra'] = ra
        session['dec'] = dec
        if glon:
            session['glon'] = glon
            session['glat'] = glat
        session['order_by'] = form_radial.order_by.data
        session['order_by_num_objs'] = form_radial.order_by_num_objs.data

    else:
        for key in ['ra', 'dec', 'glon', 'glat', 'radius']:
            if key in session:
                getattr(form_radial, key).data = session[key]
            else:
                session[key] = None

    if session['ra']:
        ra = session['ra']
        dec = session['dec']
        radius = session['radius']
        query = Candidate.query.filter(Candidate.cone_search(ra, dec, radius))
        if session['order_by'] == 'eta_residual_best':
            order_by_cond = Candidate.eta_residual_best.desc()
        elif session['order_by'] == 'eta_best':
            order_by_cond = Candidate.eta_best.asc()
        elif session['order_by'] == 'chi_squared_delta_best':
            order_by_cond = Candidate.chi_squared_delta_best.desc()

        if session['order_by_num_objs']:
            query = query.order_by(Candidate.num_objs_pass.desc(), order_by_cond)
        else:
            query = query.order_by(order_by_cond)

        page = request.args.get('page', 1, type=int)
        cands = query.paginate(page, app.config['ITEMS_PER_PAGE'], False)
        next_url = url_for('candidates', page=cands.next_num) \
            if cands.has_next else None
        prev_url = url_for('candidates', page=cands.prev_num) \
            if cands.has_prev else None
        paginate = True

        for cand in cands.items:
            sources = Source.query.filter(Source.id.in_(cand.source_id_arr)).all()
            for source in sources:
                source.load_lightcurve_plot()
    else:
        cands = None
        next_url = None
        prev_url = None
        paginate = None

    return render_template('radial_search.html', cands=cands,
                           next_url=next_url, prev_url=prev_url,
                           title='Radial Search Results', zip=zip,
                           paginate=paginate, form=form_radial)


@app.route('/reset_radial_search', methods=['GET'])
@login_required
def reset_radial_search():
    for key in ['ra', 'dec', 'glon', 'glat', 'radius']:
        try:
            del session[key]
        except KeyError:
            pass
    return redirect(url_for('radial_search'))


@app.route('/filter_search', methods=['GET', 'POST'])
@login_required
def filter_search():

    def _append_query(query, field, val_min, val_max):
        if val_min and val_max:
            if val_min > val_max:
                flash(f'{field} minimum must be less than {field} maximum', 'danger')
                return None
        if val_min:
            query = query.filter(getattr(Candidate, field) >= val_min)
        if val_max:
            query = query.filter(getattr(Candidate, field) <= val_max)
        return query

    query_fields = ['num_objs_pass', 't_0_best', 't_E_best',
                    'chi_squared_delta_best', 'rf_score_best',
                    'eta_best', 'eta_residual_best']

    form_filter = FilterSearchForm()
    if form_filter.validate_on_submit():
        for field in query_fields:
            val_min = getattr(form_filter, f'{field}_min').data
            session[f'{field}_min'] = val_min
            val_max = getattr(form_filter, f'{field}_max').data
            session[f'{field}_max'] = val_max

        session['eta_residual_slope'] = form_filter.eta_residual_slope.data
        session['order_by'] = form_filter.order_by.data
        session['order_by_num_objs'] = form_filter.order_by_num_objs.data

        flash('Filter Search Results', 'info')
    else:
        for field in query_fields:
            for minmax in ['min', 'max']:
                key = f'{field}_{minmax}'
                if f'{field}_{minmax}' in session:
                    getattr(form_filter, key).data = session[key]
                else:
                    session[key] = None
        if 'eta_residual_slope' in session:
            form_filter.eta_residual_slope.data = session['eta_residual_slope']
        else:
            session['eta_residual_slope'] = None
        for order_by_type in ['order_by', 'order_by_num_objs']:
            if order_by_type in session:
                getattr(form_filter, order_by_type).data = session[order_by_type]
            else:
                session[order_by_type] = None

    query = Candidate.query
    current_query = False
    for field in query_fields:
        val_min = session[f'{field}_min']
        val_max = session[f'{field}_max']
        query = _append_query(query, field, val_min, val_max)
        if val_min is not None or val_max is not None:
            current_query = True

    eta_residual_slope = session['eta_residual_slope']
    if eta_residual_slope:
        query = query.filter(Candidate.eta_residual_best >= Candidate.eta_best * eta_residual_slope)
        current_query = True

    if current_query:
        if session['order_by'] == 'eta_residual_best':
            order_by_cond = Candidate.eta_residual_best.desc()
        elif session['order_by'] == 'eta_best':
            order_by_cond = Candidate.eta_best.asc()
        elif session['order_by'] == 'chi_squared_delta_best':
            order_by_cond = Candidate.chi_squared_delta_best.desc()

        if session['order_by_num_objs']:
            query = query.order_by(Candidate.num_objs_pass.desc(), order_by_cond)
        else:
            query = query.order_by(order_by_cond)

        page = request.args.get('page', 1, type=int)
        cands = query.paginate(page, app.config['ITEMS_PER_PAGE'], False)
        next_url = url_for('candidates', page=cands.next_num) \
            if cands.has_next else None
        prev_url = url_for('candidates', page=cands.prev_num) \
            if cands.has_prev else None
        paginate = True

        for cand in cands.items:
            sources = Source.query.filter(Source.id.in_(cand.source_id_arr)).all()
            for source in sources:
                source.load_lightcurve_plot()
    else:
        cands = None
        next_url = None
        prev_url = None
        paginate = None

    return render_template('filter_search.html', cands=cands,
                           next_url=next_url, prev_url=prev_url,
                           title='Filter Search Results', zip=zip,
                           paginate=paginate, form=form_filter)


@app.route('/reset_filter_search', methods=['GET'])
@login_required
def reset_filter_search():
    query_fields = ['num_objs_pass', 't_0_best', 't_E_best',
                    'chi_squared_delta_best', 'rf_score_best',
                    'eta_best', 'eta_residual_best']

    for field in query_fields:
        for minmax in ['min', 'max']:
            try:
                del session[f'{field}_{minmax}']
            except KeyError:
                pass
    try:
        del session['eta_residual_slope']
    except KeyError:
        pass
    return redirect(url_for('filter_search'))
