from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, \
    SubmitField, TextAreaField, FloatField, RadioField, IntegerField
from wtforms.validators import ValidationError, DataRequired, \
    Email, EqualTo, Length, NumberRange, Optional
from puzle.models import User


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Username already taken. '
                                  'Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Email already taken. '
                                  'Please use a different email address.')


class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    about_me = TextAreaField('About me', validators=[Length(min=0, max=140)])
    submit = SubmitField('Submit')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Username already taken. '
                                      'Please use a different username.')


class ResetPasswordRequestForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('Email address not currently registered.')


class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Request Password Reset')


class EditCommentForm(FlaskForm):
    comments = TextAreaField('Comments',
                             validators=[Length(min=0, max=1024)])
    submit = SubmitField('Submit')


class RadialSearchForm(FlaskForm):
    ra = FloatField('ra (degrees)',
                    validators=[Optional(), NumberRange(min=0, max=360)])
    dec = FloatField('dec (degrees)',
                     validators=[Optional(), NumberRange(min=-90, max=90)])
    glon = FloatField('galactic longitude (degrees)',
                      validators=[Optional(), NumberRange(min=-180, max=180)])
    glat = FloatField('galactic latitude (degrees)',
                      validators=[Optional(), NumberRange(min=-90, max=90)])
    radius = FloatField('radius (arcseconds)',
                        validators=[DataRequired()])
    order_by = RadioField(choices=[('chi2', 'Order results by chi-squared ascending'),
                                   ('rchi2', 'Order results by reduced chi-squared ascending'),
                                   ('logL', 'Order results by log likelihood descending')],
                          default='rchi2')
    submit = SubmitField('Search')


class FilterSearchForm(FlaskForm):
    num_objs_pass_min = IntegerField('num_objs_pass min',
                                   validators=[Optional(), NumberRange(min=1)])
    num_objs_pass_max = IntegerField('num_objs_pass max',
                                   validators=[Optional(), NumberRange(min=1)])
    rf_score_best_min = FloatField('rf_score_best min',
                                   validators=[Optional(), NumberRange(min=0, max=1)])
    rf_score_best_max = FloatField('rf_score_best max',
                                   validators=[Optional(), NumberRange(min=0, max=1)])
    eta_best_min = FloatField('eta_best min',
                              validators=[Optional(), NumberRange(min=0)])
    eta_best_max = FloatField('eta_best max',
                              validators=[Optional(), NumberRange(min=0)])
    minmax_t_0_best_min = FloatField('minmax: t_0_best min (days)',
                                     validators=[Optional(), NumberRange(min=58144)])
    minmax_t_0_best_max = FloatField('minmax: t_0_best max (days)',
                                     validators=[Optional(), NumberRange(max=59293)])
    minmax_t_E_best_min = FloatField('minmax: t_E_best min (days)',
                                     validators=[Optional(), NumberRange(min=0.01)])
    minmax_t_E_best_max = FloatField('minmax: t_E_best max (days)',
                                     validators=[Optional(), NumberRange(max=5000)])
    minmax_chi_squared_delta_best_min = FloatField('minmax: chi_squared_delta_best min',
                                                   validators=[Optional(), NumberRange(min=0)])
    minmax_chi_squared_delta_best_max = FloatField('minmax: chi_squared_delta_best max',
                                                   validators=[Optional(), NumberRange(min=0)])
    minmax_eta_residual_best_min = FloatField('minmax: eta_residual_best min',
                                              validators=[Optional(), NumberRange(min=0)])
    minmax_eta_residual_best_max = FloatField('minmax: eta_residual_best max',
                                              validators=[Optional(), NumberRange(min=0)])
    opt_t0_best_min = FloatField('opt: t0_best min (days)',
                                  validators=[Optional(), NumberRange(min=0)])
    opt_t0_best_max = FloatField('opt: t0_best max (days)',
                                  validators=[Optional(), NumberRange(min=0)])
    opt_tE_best_min = FloatField('opt: tE_best min (days)',
                                  validators=[Optional(), NumberRange(min=0)])
    opt_tE_best_max = FloatField('opt: tE_best max (days)',
                                  validators=[Optional(), NumberRange(min=0)])
    opt_chi_squared_ulens_best_min = FloatField('opt: chi_squared_ulens_best min',
                                                validators=[Optional(), NumberRange(min=0)])
    opt_chi_squared_ulens_best_max = FloatField('opt: chi_squared_ulens_best max',
                                                validators=[Optional(), NumberRange(min=0)])
    opt_eta_residual_best_min = FloatField('opt: eta_residual_best min',
                                           validators=[Optional(), NumberRange(min=0)])
    opt_eta_residual_best_max = FloatField('opt: eta_residual_best max',
                                           validators=[Optional(), NumberRange(min=0)])
    order_by = RadioField(choices=[('eta_best', 'Order results by eta ascending'),
                                   ('minmax_eta_residual_best', 'Order results by minmax eta_residual descending'),
                                   ('minmax_chi_squared_delta_best', 'Order results by minmax chi_squared_delta desc'),
                                   ('opt_eta_residual_best', 'Order results by opt eta_residual descending'),
                                   ('opt_chi_squared_ulens_best', 'Order results optmax chi_squared_ulens desc')],
                          default='opt_eta_residual_best')
    order_by_num_objs = BooleanField('Order by number of objects descending', default=True)
    submit = SubmitField('Search')


class EmptyForm(FlaskForm):
    submit = SubmitField('Submit')
