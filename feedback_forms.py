import re

from flask_wtf import FlaskForm
from wtforms import BooleanField, HiddenField, SelectField, StringField, TextAreaField
from wtforms.validators import DataRequired, Length, Optional, ValidationError


FEEDBACK_CATEGORIES = [
    ("general_feedback", "General feedback"),
    ("confusing_language", "Confusing page or language"),
    ("incorrect_source", "Incorrect hazard or source information"),
    ("organization_demo", "Organization or demo interest"),
    ("partnership_sponsorship", "Partnership or sponsorship interest"),
    ("other_question", "Other question"),
]
ORGANIZATION_INTEREST_TYPES = [
    ("feedback_call", "Feedback call"),
    ("demo", "Product demo"),
    ("pilot", "Community pilot"),
    ("sponsorship", "Sponsorship discussion"),
    ("partnership", "Partnership conversation"),
]
UPDATE_USER_TYPES = [
    ("", "Select one (optional)"),
    ("resident", "Resident"),
    ("parent", "Parent or caregiver"),
    ("student", "Student"),
    ("educator", "Educator"),
    ("nonprofit_community", "Nonprofit or community organization"),
    ("cert_volunteer", "CERT or preparedness volunteer"),
    ("other", "Other"),
]
SAFE_PAGE_CONTEXTS = {
    "feedback",
    "home",
    "map",
    "privacy",
    "risk_summary",
    "sources",
    "terms",
}
EMAIL_REQUIRED_CATEGORIES = {
    "organization_demo",
    "partnership_sponsorship",
    "other_question",
}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_CITY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]*(?:,\s*[A-Za-z]{2})?$")


def _reject_control_characters(form, field):
    if field.data and _CONTROL_CHARACTERS.search(str(field.data)):
        raise ValidationError("Remove unsupported control characters.")


def _validate_optional_email(form, field):
    value = (field.data or "").strip().lower()
    if value and (len(value) > 254 or not _EMAIL_PATTERN.fullmatch(value)):
        raise ValidationError("Enter a valid email address.")


def _validate_safe_page_context(form, field):
    if field.data not in SAFE_PAGE_CONTEXTS:
        raise ValidationError("Invalid page context.")


def _validate_zip_or_city(form, field):
    value = (field.data or "").strip()
    if not value:
        return
    if re.fullmatch(r"\d{5}", value) or _CITY_PATTERN.fullmatch(value):
        return
    raise ValidationError("Enter a five-digit ZIP code or city name, not a street address.")


class FeedbackForm(FlaskForm):
    form_kind = HiddenField(default="feedback")
    page_context = HiddenField(validators=[DataRequired(), _validate_safe_page_context])
    category = SelectField("What would you like to share?", choices=FEEDBACK_CATEGORIES)
    name = StringField(
        "Name (optional)",
        validators=[Optional(), Length(max=100), _reject_control_characters],
    )
    email = StringField(
        "Email",
        validators=[Length(max=254), _validate_optional_email],
    )
    message = TextAreaField(
        "Message",
        validators=[DataRequired(), Length(min=10, max=2000), _reject_control_characters],
    )
    website = StringField("Website", validators=[Optional(), Length(max=200)])

    def validate_email(self, field):
        if self.category.data in EMAIL_REQUIRED_CATEGORIES and not (field.data or "").strip():
            raise ValidationError("Email is required for questions and follow-up requests.")


class OrganizationInterestForm(FlaskForm):
    form_kind = HiddenField(default="organization")
    page_context = HiddenField(validators=[DataRequired(), _validate_safe_page_context])
    name = StringField(
        "Name",
        validators=[DataRequired(), Length(max=100), _reject_control_characters],
    )
    organization = StringField(
        "Organization",
        validators=[DataRequired(), Length(max=160), _reject_control_characters],
    )
    role = StringField(
        "Role",
        validators=[DataRequired(), Length(max=100), _reject_control_characters],
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Length(max=254), _validate_optional_email],
    )
    interest_type = SelectField(
        "What are you interested in?",
        choices=ORGANIZATION_INTEREST_TYPES,
    )
    message = TextAreaField(
        "How might your community use StayReady?",
        validators=[DataRequired(), Length(min=10, max=2000), _reject_control_characters],
    )
    website = StringField("Website", validators=[Optional(), Length(max=200)])


class UpdateInterestForm(FlaskForm):
    form_kind = HiddenField(default="updates")
    email = StringField(
        "Email",
        validators=[DataRequired(), Length(max=254), _validate_optional_email],
    )
    location = StringField(
        "ZIP or city (optional)",
        validators=[
            Optional(),
            Length(max=100),
            _reject_control_characters,
            _validate_zip_or_city,
        ],
    )
    user_type = SelectField("I am a...", choices=UPDATE_USER_TYPES, validators=[Optional()])
    consent = BooleanField(
        "I agree to receive StayReady product updates and preparedness resources.",
        validators=[DataRequired()],
    )
    website = StringField("Website", validators=[Optional(), Length(max=200)])
