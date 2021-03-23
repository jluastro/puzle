from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, \
    SubmitField, TextAreaField, FloatField, RadioField
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


class EditSourceCommentForm(FlaskForm):
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
    order_by = RadioField(choices=[('eta_best', 'Order results by eta ascending'),
                                  ('chi_squared_delta_best', 'Order results by chi_squared_delta desc')],
                          default='eta_best')
    submit = SubmitField('Search')


class FilterSearchForm(FlaskForm):
    num_objs_pass_min = FloatField('num_objs_pass min',
                                   validators=[Optional(), NumberRange(min=1)])
    num_objs_pass_max = FloatField('num_objs_pass max',
                                   validators=[Optional(), NumberRange(min=1)])
    t_E_best_min = FloatField('t_E_best min (days)',
                              validators=[Optional(), NumberRange(min=0)])
    t_E_best_max = FloatField('t_E_best max (days)',
                              validators=[Optional(), NumberRange(min=0)])
    chi_squared_delta_best_min = FloatField('chi_squared_delta_best min',
                                            validators=[Optional(), NumberRange(min=0)])
    chi_squared_delta_best_max = FloatField('chi_squared_delta_best max',
                                            validators=[Optional(), NumberRange(min=0)])
    rf_score_best_min = FloatField('rf_score_best min',
                                   validators=[Optional(), NumberRange(min=0)])
    rf_score_best_max = FloatField('rf_score_best max',
                                   validators=[Optional(), NumberRange(min=0)])
    order_by = RadioField(choices=[('eta_best', 'Order results by eta ascending'),
                                  ('chi_squared_delta_best', 'Order results by chi_squared_delta desc')],
                          default='eta_best')
    submit = SubmitField('Search')


class CandidateOrderForm(FlaskForm):
    order_by = RadioField(choices=[('eta_best', 'Re-order candidates by eta ascending'),
                                  ('chi_squared_delta_best', 'Re-order candidates by chi_squared_delta desc')],
                          default='eta_best')
    submit = SubmitField('Order By')


class EmptyForm(FlaskForm):
    submit = SubmitField('Submit')
