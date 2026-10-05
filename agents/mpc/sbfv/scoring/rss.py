"""Resilience Skill Score from integer-cent costs (design §7.1; Q9, Q55, Q61).

(55) RSS_s = sum_{n in s} g_n / sum_{n in s} D_n with g_n = J^naive_n - J^pi_n and D_n = J^naive_n - J^oracle_n.
(56) RSS_G = sum_s p_s g-bar_s / sum_s p_s D-bar_s with the fixed harm-quantile weights p_s = (0.50, 0.30, 0.15, 0.05).
Computed exactly from integers (``fractions.Fraction``) so naive scores exactly 0 and a policy costing J^oracle exactly
1 (V30). Weights are read as decimals, ``Fraction(str(w))``, so 0.30 is exactly 3/10. Negative scores are allowed and
never clipped.
"""

from fractions import Fraction
from numbers import Integral
from typing import Sequence


STRATUM_WEIGHTS = (0.50, 0.30, 0.15, 0.05)  # p_s of (44), SYNTHETIC cut points (Q61)


def _int_cents(xs: Sequence[int], name: str) -> list[int]:
    """Integer cents (24) as Python ints; floats, booleans and anything else non-integral are refused."""
    out = []
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, Integral):
            raise ValueError(f"{name}: costs must be integer cents (24), got {x!r} of type {type(x).__name__}")
        out.append(int(x))
    return out


def _savings(J_pi: Sequence[int], J_naive: Sequence[int], J_oracle: Sequence[int]) -> tuple[list[int], list[int]]:
    """Per-episode g_n = J^naive_n - J^pi_n and D_n = J^naive_n - J^oracle_n, after validating the inputs."""
    pi, naive, oracle = _int_cents(J_pi, "J_pi"), _int_cents(J_naive, "J_naive"), _int_cents(J_oracle, "J_oracle")
    if not len(pi) == len(naive) == len(oracle):
        raise ValueError(f"lengths differ: J_pi {len(pi)}, J_naive {len(naive)}, J_oracle {len(oracle)}")
    g = [a - b for a, b in zip(naive, pi)]
    D = [a - b for a, b in zip(naive, oracle)]
    return g, D


def rss_stratum(J_pi: Sequence[int], J_naive: Sequence[int], J_oracle: Sequence[int]) -> float:
    """(55) over the episodes of one stratum; integer cents in, a float out.

    Args:
        J_pi: the policy's episode costs J^pi_n in integer cents.
        J_naive: the naive anchor's costs J^naive_n in integer cents, same episodes.
        J_oracle: the oracle's costs J^oracle_n in integer cents, same episodes.

    Returns:
        The exact ratio sum g_n / sum D_n, rounded once to the nearest float.

    Raises:
        ValueError: if the denominator sum of D_n is not positive, the lengths differ, or a cost is not an integer.

    """
    g, D = _savings(J_pi, J_naive, J_oracle)
    den = sum(D)
    if den <= 0:
        raise ValueError(f"(55) denominator sum of D_n = {den} cents is not positive")
    return float(Fraction(sum(g), den))


def rss_pooled(
    J_pi: Sequence[int],
    J_naive: Sequence[int],
    J_oracle: Sequence[int],
    strata: Sequence[int],
    weights: Sequence[float] = STRATUM_WEIGHTS,
) -> float:
    """(56): strata are 1-based stratum labels per episode; every stratum with a weight must have N_s >= 1.

    Args:
        J_pi: the policy's episode costs J^pi_n in integer cents.
        J_naive: the naive anchor's costs J^naive_n in integer cents, same episodes.
        J_oracle: the oracle's costs J^oracle_n in integer cents, same episodes.
        strata: the stratum s(omega_n) of (44) per episode, an integer in 1..len(weights).
        weights: p_s per stratum, read as decimals (``Fraction(str(w))``); non-negative. A stratum of weight 0 may be
            empty and contributes nothing.

    Returns:
        The exact ratio sum_s p_s g-bar_s / sum_s p_s D-bar_s, rounded once to the nearest float.

    Raises:
        ValueError: if the lengths differ, a cost or label is not an integer, a label lies outside 1..len(weights), a
            weight is negative, a stratum with positive weight has no episode, or the denominator is not positive.

    """
    g, D = _savings(J_pi, J_naive, J_oracle)
    labels = list(strata)
    if len(labels) != len(g):
        raise ValueError(f"lengths differ: {len(g)} costs, {len(labels)} stratum labels")
    p = [Fraction(str(w)) for w in weights]
    if not p or any(w < 0 for w in p):
        raise ValueError(f"weights must be a non-empty sequence of non-negative numbers, got {tuple(weights)!r}")
    n_strata = len(p)
    g_sum, D_sum, N = [0] * n_strata, [0] * n_strata, [0] * n_strata
    for s, gn, Dn in zip(labels, g, D):
        if isinstance(s, bool) or not isinstance(s, Integral) or not 1 <= s <= n_strata:
            raise ValueError(f"stratum label {s!r} is not an integer in 1..{n_strata}")
        g_sum[s - 1] += gn
        D_sum[s - 1] += Dn
        N[s - 1] += 1
    num = den = Fraction(0)
    for i in range(n_strata):
        if p[i] == 0:
            continue
        if N[i] == 0:
            raise ValueError(f"stratum {i + 1} has weight {p[i]} and no episode (N_s >= 1 required by (56))")
        num += p[i] * Fraction(g_sum[i], N[i])
        den += p[i] * Fraction(D_sum[i], N[i])
    if den <= 0:
        raise ValueError(f"(56) denominator sum_s p_s D-bar_s = {float(den)} cents is not positive")
    return float(num / den)
