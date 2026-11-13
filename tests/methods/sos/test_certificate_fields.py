"""Every SOS certificate field has an explicit replay role."""

from dataclasses import fields

from symbopt import SOSCertificate

FIELD_ROLES = {
    "polynomial": "statement",
    "variables": "coordinate_scope",
    "lower_bound": "statement",
    "gram_matrices": "witness",
    "gram_bases": "witness",
    "gram_multipliers": "domain_premise",
    "equalities": "domain_premise",
    "eq_multiplier_polys": "witness",
    "normalization_scale": "statement",
    "residual": "derived_summary",
    "exact_psd_checks": "derived_summary",
    "exact_residual_zero": "derived_summary",
}


def test_field_roles():
    assert {field.name for field in fields(SOSCertificate)} == set(FIELD_ROLES)
