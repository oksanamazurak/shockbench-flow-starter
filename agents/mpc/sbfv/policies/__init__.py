"""Baselines (design §8): `zero` and the naive anchor in milestone M1; the suite in M4."""

from sbfv.policies.base import Policy, ZeroPolicy, empty_action
from sbfv.policies.naive import NaiveFallback, NaivePolicy


__all__ = ["NaiveFallback", "NaivePolicy", "Policy", "ZeroPolicy", "empty_action"]
