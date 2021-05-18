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
    num_objs_pass_min = IntegerField('num_objs_pass Level2 min',
                                   validators=[Optional(), NumberRange(min=1)])
    num_objs_pass_max = IntegerField('num_objs_pass Level2 max',
                                   validators=[Optional(), NumberRange(min=1)])
    t0_pspl_gp_min = FloatField('t0_pspl_gp min (days)',
                                validators=[Optional(), NumberRange(min=57829)])
    t0_pspl_gp_max = FloatField('t0_pspl_gp max (days)',
                                validators=[Optional(), NumberRange(max=59608)])
    tE_pspl_gp_min = FloatField('tE_pspl_gp min (days)',
                                validators=[Optional(), NumberRange(min=0.01)])
    tE_pspl_gp_max = FloatField('tE_pspl_gp max (days)',
                                validators=[Optional(), NumberRange(max=10000)])
    u0_amp_pspl_gp_min = FloatField('u0_amp_pspl_gp min (days)',
                                    validators=[Optional(), NumberRange(min=-2)])
    u0_amp_pspl_gp_max = FloatField('u0_amp_pspl_gp max (days)',
                                    validators=[Optional(), NumberRange(max=2)])
    piE_pspl_gp_min = FloatField('piE_pspl_gp min (days)',
                                 validators=[Optional(), NumberRange(min=-2)])
    piE_pspl_gp_max = FloatField('piE_pspl_gp max (days)',
                                 validators=[Optional(), NumberRange(max=2)])
    b_sff1_pspl_gp_min = FloatField('b_sff1_pspl_gp min (days)',
                                    validators=[Optional(), NumberRange(min=0)])
    b_sff1_pspl_gp_max = FloatField('b_sff1_pspl_gp max (days)',
                                    validators=[Optional(), NumberRange(max=2)])
    mag_base1_pspl_gp_min = FloatField('mag_base1_pspl_gp min (days)',
                                       validators=[Optional(), NumberRange(min=15)])
    mag_base1_pspl_gp_max = FloatField('mag_base1_pspl_gp max (days)',
                                       validators=[Optional(), NumberRange(max=25)])
    chi2_pspl_gp_min = FloatField('chi2_pspl_gp min (days)',
                                  validators=[Optional(), NumberRange(min=0)])
    chi2_pspl_gp_max = FloatField('chi2_pspl_gp max (days)',
                                  validators=[Optional(), NumberRange(max=10000)])
    rchi2_pspl_gp_min = FloatField('rchi2_pspl_gp min (days)',
                                   validators=[Optional(), NumberRange(min=0)])
    rchi2_pspl_gp_max = FloatField('rchi2_pspl_gp max (days)',
                                   validators=[Optional(), NumberRange(max=100)])
    logL_pspl_gp_min = FloatField('logL_pspl_gp min (days)',
                                   validators=[Optional(), NumberRange(min=0)])
    logL_pspl_gp_max = FloatField('logL_pspl_gp max (days)',
                                   validators=[Optional(), NumberRange(max=100)])
    order_by = RadioField(choices=[('chi2', 'Order results by chi-squared ascending'),
                                   ('rchi2', 'Order results by reduced chi-squared ascending'),
                                   ('logL', 'Order results by log likelihood descending'),
                                   ('tE', 'Order results by tE desc')],
                          default='rchi2')
    submit = SubmitField('Search')


class EmptyForm(FlaskForm):
    submit = SubmitField('Submit')


class CategorizeForm(FlaskForm):
    category_return = RadioField(choices=[('random', 'Randomly select new candidate after categorization'),
                                          ('same', 'Return to same candidate after categorization')],
                                 default='random')
    submit = SubmitField('Submit')


class BrowseForm(FlaskForm):
    category_none = BooleanField('no category', default=True)
    category_clear_microlensing = BooleanField('clear microlensing', default=True)
    category_possible_microlensing = BooleanField('possible microlensing', default=True)
    category_no_variability = BooleanField('no variability', default=True)
    category_poor_model_fit = BooleanField('poor model fit', default=True)
    category_non_microlensing_variable = BooleanField('non-microlensing variable', default=True)
    submit = SubmitField('Submit')
