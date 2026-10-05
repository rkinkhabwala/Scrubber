"""Canonical PII labels and the mapping from ai4privacy source labels."""
from __future__ import annotations

LABELS: dict[str, str] = {
    "NAME": "a person's first, last or full name",
    "EMAIL": "an email address",
    "PHONE": "a phone or fax number",
    "ADDRESS": "a street address, building number, city or ZIP/postal code",
    "DATE": "a calendar date tied to a person (birth date, visit date, etc.)",
    "GOV_ID": "SSN, tax ID, passport, national ID or driver's license number",
    "CARD": "a payment card number",
    "ACCOUNT": "a bank account, member ID, medical record number or policy number",
    "AGE": "a person's age",
}

SOURCE_TO_LABEL: dict[str, str | None] = {
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "DATE": "DATE",
    "SOCIALNUM": "GOV_ID",
    "TAXNUM": "GOV_ID",
    "PASSPORTNUM": "GOV_ID",
    "IDCARDNUM": "GOV_ID",
    "DRIVERLICENSENUM": "GOV_ID",
    "CREDITCARDNUMBER": "CARD",
    "AGE": "AGE",
    # Deliberately not treated as PII in this project:
    "TIME": None,
    "SEX": None,
    "TITLE": None,
    "ORGANISATIONPLACEHOLDER": None,
}

# Labels whose adjacent spans (separated only by spaces) should be merged into one entity.
MERGE_ADJACENT = {"NAME"}
