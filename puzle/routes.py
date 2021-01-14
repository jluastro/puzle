from flask import render_template, flash, redirect, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from datetime import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u
from puzle import app, db
from puzle.forms import LoginForm, RegistrationForm, \
    EditProfileForm, ResetPasswordRequestForm, ResetPasswordForm, \
    EditSourceCommentForm, SearchForm, EmptyForm
from puzle.models import User, Source
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
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    sources = user.followed_sources().paginate(page, app.config['SOURCES_PER_PAGE'], False)
    next_url = url_for('user', username=username, page=sources.next_num) \
        if sources.has_next else None
    prev_url = url_for('user', username=username, page=sources.prev_num) \
        if sources.has_prev else None
    return render_template('user.html', user=user, sources=sources,
                           next_url=next_url, prev_url=prev_url)


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
    form = EmptyForm()
    source = Source.query.filter_by(id=sourceid).first_or_404()
    source.load_lightcurve_plot()
    return render_template('source.html', source=source,
                           form=form)


@app.route('/edit_source_comments/<sourceid>', methods=['GET', 'POST'])
@login_required
def edit_source_comments(sourceid):
    source = Source.query.filter_by(id=sourceid).first_or_404()
    form = EditSourceCommentForm()
    if form.validate_on_submit():
        source.comments = form.comments.data
        db.session.commit()
        flash('Your changes have been saved.', 'success')
        return redirect(url_for('source', sourceid=sourceid))
    elif request.method == 'GET':
        form.comments.data = source.comments
    return render_template('edit_source_comments.html',
                           form=form)


@app.route('/fetch_ztf_ids/<sourceid>', methods=['POST'])
@login_required
def fetch_ztf_ids(sourceid):
    source = Source.query.filter_by(id=sourceid).first_or_404()
    n_ids = source.fetch_ztf_ids()
    flash('%i ZTF IDs Found' % n_ids, 'success')
    db.session.commit()
    return redirect(url_for('source', sourceid=sourceid))


@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    form = SearchForm()
    if form.validate_on_submit():
        if form.ra.data and form.dec.data:
            ra, dec = form.ra.data, form.dec.data
            flash('Searching (ra,dec) = (%.5f, %.5f)' % (ra, dec),
                  'info')
        elif form.glon.data and form.glat.data:
            glon, glat = form.glon.data, form.glat.data
            flash('Searching (glon, glat) = (%.5f, %.5f)' % (glon, glat),
                  'info')
            coord = SkyCoord(glon, glat,
                             unit=u.degree, frame='galactic')
            ra = coord.icrs.ra.value
            dec = coord.icrs.dec.value
        else:
            flash('Either (ra, dec) or '
                  '(glon, glat) '
                  'must be entered.', 'danger')
            return redirect(url_for('search'))

        sources = db.session.query(Source).filter(
            Source.cone_search(ra, dec, form.radius.data)).all()
        sources.sort(key=lambda x: x.id)
        return render_template('search.html', form=form, sources=sources)
    return render_template('search.html', form=form)


@app.route('/follow/<sourceid>', methods=['POST'])
@login_required
def follow(sourceid):
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


@app.route('/unfollow/<sourceid>', methods=['POST'])
@login_required
def unfollow(sourceid):
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


@app.route('/sources', methods=['GET', 'POST'])
@login_required
def sources():
    page = request.args.get('page', 1, type=int)
    sources = Source.query.order_by(Source.id.asc()).paginate(page, app.config['SOURCES_PER_PAGE'], False)
    next_url = url_for('sources', page=sources.next_num) \
        if sources.has_next else None
    prev_url = url_for('sources', page=sources.prev_num) \
        if sources.has_prev else None
    return render_template('sources.html', sources=sources,
                           next_url=next_url, prev_url=prev_url)
