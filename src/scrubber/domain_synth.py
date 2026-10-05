"""Synthetic in-domain examples with exact character spans.

Templates mimic the text a bank, insurer or software company actually needs to redact: support
tickets, claim notes, application logs and emails. Each template also contains HARD NEGATIVES —
things that look sensitive but are not personal data (transaction IDs, CPT codes, amounts, company
names) — so the model learns what NOT to redact.

All values are fake (Faker). Swap in your own templates to adapt Scrubber to your company.
"""
from __future__ import annotations

import random
import re

from faker import Faker

# {slot:LABEL} is PII; {slot} without a label is filler (not PII).
TEMPLATES = [
    # --- bank support tickets
    "Ticket {ticket}: Customer {name:NAME} called about a declined charge of {amount} at {company}. "
    "Card on file {card:CARD}, acct {acct:ACCOUNT}. Callback {phone:PHONE}.",
    "Hi team, {name:NAME} (DOB {dob:DATE}) says the wire {txn} for {amount} never arrived. "
    "Their email is {email:EMAIL}. Please check routing for account {acct:ACCOUNT}.",
    "Chat transcript #{ticket}\nAgent: Can you confirm your address?\nCustomer: {street:ADDRESS}, "
    "{city:ADDRESS} {zip:ADDRESS}\nAgent: Thanks {first:NAME}, I've reset the card ending {last4}.",
    "Escalation: dispute on txn {txn} ({amount}, merchant {company}). Cardholder {name:NAME}, "
    "SSN on file {ssn:GOV_ID}. Fraud team please review by {due}.",
    # --- insurance / healthcare claim notes
    "Claim {claim} for member {name:NAME}, member ID {member:ACCOUNT}, DOS {dos:DATE}. "
    "CPT {cpt} billed {amount}; denied for missing auth. Provider {company}.",
    "Pt {name:NAME}, {age:AGE} y/o, MRN {mrn:ACCOUNT}, seen {dos:DATE} for follow-up. Code {cpt}. "
    "Contact {phone:PHONE}. Copay {amount} collected.",
    "Appeal received from {name:NAME} ({email:EMAIL}) re: claim {claim}. Policy {policy:ACCOUNT}. "
    "Mail decision to {street:ADDRESS}, {city:ADDRESS} {zip:ADDRESS}.",
    # --- application logs
    "{ts} INFO  auth-svc login ok user={email:EMAIL} ip_hash={hash} req={txn}",
    "{ts} WARN  payments retry=3 order={ticket} card={card:CARD} amount={amount} gateway={company}",
    "{ts} ERROR kyc-svc verification failed name=\"{name:NAME}\" dob={dob:DATE} ssn={ssn:GOV_ID} trace={hash}",
    "{ts} DEBUG profile update uid={uid} phone={phone:PHONE} zip={zip:ADDRESS} build={build}",
    # --- emails
    "Subject: Update on case {ticket}\n\nDear {name:NAME},\nWe received your documents, including "
    "driver's license {dl:GOV_ID}. Your refund of {amount} will post within 5 business days.\n"
    "Best,\n{agent:NAME}\n{company} Support",
    "From: {email:EMAIL}\nTo: support@{domain}\nMy name is {name:NAME}, born {dob:DATE}. "
    "I moved to {street:ADDRESS}, {city:ADDRESS}. Please update my policy {policy:ACCOUNT}.",
    # --- no-PII negatives (the model must return an empty list)
    "{ts} INFO  scheduler job={txn} finished in 412ms build={build}",
    "Ticket {ticket}: {company} reports CPT {cpt} reimbursement changed to {amount} effective next quarter.",
    "Reminder: release {build} ships Friday. Ticket {ticket} tracks the {company} integration.",
]

# Never used for training: test_domain mixes these in so the score reflects new sentence shapes,
# not memorized templates.
HELDOUT_TEMPLATES = [
    "Voicemail from {name:NAME} at {phone:PHONE}: says card {card:CARD} was charged twice by {company} "
    "for {amount}. Ref {ticket}.",
    "{ts} INFO  onboarding step=3 applicant=\"{name:NAME}\" email={email:EMAIL} dl={dl:GOV_ID} req={txn}",
    "Prior auth request {claim}: {name:NAME}, DOB {dob:DATE}, member {member:ACCOUNT}, CPT {cpt}, "
    "lives at {street:ADDRESS}, {city:ADDRESS} {zip:ADDRESS}.",
    "Weekly report: {company} processed {amount} in refunds; tickets {ticket} and {txn} remain open.",
]

_SLOT = re.compile(r"\{(\w+)(?::(\w+))?\}")


def _value(fake: Faker, rng: random.Random, slot: str) -> str:
    gen = {
        "name": lambda: fake.name(),
        "first": lambda: fake.first_name(),
        "agent": lambda: fake.name(),
        "email": lambda: fake.email(),
        "phone": lambda: fake.phone_number(),
        "card": lambda: fake.credit_card_number(),
        "acct": lambda: str(rng.randint(10**9, 10**12 - 1)),
        "member": lambda: f"{rng.choice('ABCHKMW')}{rng.randint(10**8, 10**9 - 1)}",
        "mrn": lambda: f"MRN-{rng.randint(10**6, 10**7 - 1)}",
        "policy": lambda: f"POL-{rng.randint(10**6, 10**7 - 1)}-{rng.choice('ABC')}",
        "ssn": lambda: fake.ssn(),
        "dl": lambda: f"{rng.choice('ABCDEFG')}{rng.randint(10**7, 10**8 - 1)}",
        "dob": lambda: fake.date_of_birth(minimum_age=18, maximum_age=90).strftime(
            rng.choice(["%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y"])),
        "dos": lambda: fake.date_this_year().strftime(rng.choice(["%m/%d/%Y", "%Y-%m-%d"])),
        "age": lambda: str(rng.randint(19, 88)),
        "street": lambda: fake.street_address(),
        "city": lambda: fake.city(),
        "zip": lambda: fake.zipcode(),
        # --- not PII
        "ticket": lambda: f"{rng.choice(['INC', 'CS', 'TKT'])}-{rng.randint(10000, 99999)}",
        "txn": lambda: f"TXN{rng.randint(10**9, 10**10 - 1)}",
        "claim": lambda: f"CLM-{rng.randint(10**6, 10**7 - 1)}",
        "amount": lambda: f"${rng.randint(5, 4000)}.{rng.randint(0, 99):02d}",
        "company": lambda: fake.company(),
        "cpt": lambda: rng.choice(["99213", "99214", "80053", "85025", "71046", "93000", "97110"]),
        "due": lambda: rng.choice(["Friday", "EOD", "next week"]),
        "last4": lambda: str(rng.randint(1000, 9999)),
        "ts": lambda: fake.date_time_this_year().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hash": lambda: fake.sha1()[:12],
        "uid": lambda: fake.uuid4()[:8],
        "build": lambda: f"v{rng.randint(1, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}",
        "domain": lambda: fake.domain_name(),
    }[slot]
    return gen()


def fill(template: str, fake: Faker, rng: random.Random) -> dict:
    text, spans, cursor = [], [], 0
    pos = 0
    reuse: dict[str, str] = {}  # the same slot name in one template gets the same value
    for m in _SLOT.finditer(template):
        literal = template[cursor:m.start()]
        text.append(literal)
        pos += len(literal)
        slot, label = m.group(1), m.group(2)
        value = reuse.setdefault(slot, _value(fake, rng, slot))
        if label:
            spans.append({"start": pos, "end": pos + len(value), "label": label, "value": value})
        text.append(value)
        pos += len(value)
        cursor = m.end()
    text.append(template[cursor:])
    return {"text": "".join(text), "spans": spans, "source": "domain"}


def generate(n: int, seed: int = 7, templates: list[str] | None = None) -> list[dict]:
    templates = templates or TEMPLATES
    rng = random.Random(seed)
    fake = Faker(["en_US"])
    fake.seed_instance(seed)
    return [fill(templates[i % len(templates)], fake, rng) for i in range(n)]
