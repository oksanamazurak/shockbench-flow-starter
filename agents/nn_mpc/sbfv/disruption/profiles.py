"""Generator profiles: ``GeneratorParams`` for a given instance (design §2.4 "`tiny` generator profile", milestone M2).

``tiny_profile(inst)`` returns the profile table of §2.4 as data. Every value there is SYNTHETIC(placeholder) unless
tagged; the params tree carries the tags in field metadata (``params.provenance`` lists them by leaf).

The militarised baseline of the region(s) adjacent to the chokepoint is calibrated so that the stationary
militarised-closure rate on `chk` at the anchor rung is 0.43 x 4/7 per year (Q52 record scaled: `chk` stands for four
of the seven flagship chokepoints), by (32) with the regime multipliers at their stationary means (``closure_rate``):
both are expectations over the stationary law pi^c of the weekly conflict chain Pi^c of (28), the militarised
multiplier E[m_M(z^c)] and the policy multiplier E[(1 - P(tension | z^c)) m_P(normal) + P(tension | z^c) m_P(tension)],
with P(tension | z^c) of (28) (``regime.tension_stationary``; u_p is the share of time in the normal state, so
P(tension | none) = 1 - u_p). The type law's closure share is averaged over pi^c too (a regional conflict is feasible
only in war), and the closure modulation g(X_c)/c_X has mean 1 by the choice of c_X (29). ``tiny_profile`` runs that
calibration (``calibrate_closure_baselines``) whenever it builds the profile: it is deterministic on one machine, and
its LAPACK calls (eig, solve, eigvals) may round the last bits differently on another, which Q94 accepts (bit-identity
across machines is not pursued).

The burn-in bound (38), B_burn >= t_95 + max_ty F^{-1}_{T_q|ty}(0.99), is checked over the rungs of the family
(``check_burn_in``, which ``tiny_profile`` runs): t_95 of (33) at each rung under the block Gamma of (31) and the
params' 1/beta; the quantile over every event type the generator draws, the sanction and material-outage tail
included, a regional conflict by its war-profile window |D_q| = max(T_q, w) + w of (37), (40) (§4.3) and a port strike
by its stoppage/slowdown mixture.

``flagship_profile(inst)`` is the profile of `small` and `full` (milestone M5; design §12 M5 stream "generator" rows):
`tiny`'s table with the TIES Weibull for sanctions and material outages (Q106), per-region rates held at `full`'s,
Gamma^MP 0.01 (the owner's M5-O7 answer), the dyad strait closure (M5-O13), the one-scale (32) calibration to 0.43
closures per year on all seven chokepoints over the instance's chokepoint adjacency (M5-O8 (b)), with each region's
effective conflict law (dyad wars included) and the strait closures counted (``flagship_closure_rates``), and the
burn-in of Q97's 1 % rule (``burn_in_family``, Q106). The owner-level values it reads are the named constants of the
M5 block (``FLAGSHIP_RATE_REGIONS``, ``BURN_IN_TYPES``, ``FLAGSHIP_DYADS``, ``CLOSURE_SPLIT``) and the flagship
subclasses of ``params``.
"""

import dataclasses
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from sbfv.disruption.events import _strike_regions as strike_regions
from sbfv.disruption.hawkes import (
    M_BLOCK,
    P_BLOCK,
    branching_matrix,
    relaxation_time,
    spectral_radius,
    stationary_rates,
)
from sbfv.disruption.laws import Law, mixture_quantile
from sbfv.disruption.params import (
    RUNGS,
    TIES_WEIBULL,
    WEEKS_PER_YEAR,
    FlagshipGeneratorParams,
    FlagshipHawkesParams,
    FlagshipMarkLaws,
    GeneratorParams,
    HawkesParams,
    RegimeParams,
    TargetingRules,
)
from sbfv.disruption.regime import WAR, tension_stationary, weekly_conflict_matrix
from sbfv.disruption.targets import NOOP, adjacency_dict, strait_chokepoints, targeting_rules, type_weights
from sbfv.instance.schema import Instance
from sbfv.marks import MILITARISED_CLOSURE, REGIONAL_CONFLICT, war_windows
from sbfv.omega import codes


ANCHOR_EVENTS_PER_YEAR = 12.0  # SYNTHETIC(prior: scripts/python/evidence/budget_rejection.py)
ANCHOR_GAMMA = RUNGS[0]
POLICY_SHARE = 0.70  # SYNTHETIC(prior: scripts/python/evidence/f5_realism.py)
FLAGSHIP_CLOSURES_PER_YEAR = 0.43  # DERIVED(W2-maritime §2.2 event list), the anchor of Q52 / Q76 owner D5
TINY_CHOKEPOINT_SHARE = 4 / 7  # `chk` stands for Hormuz, Malacca, Suez/Red Sea and the Cape reroute (§2.4)
BURN_IN_QUANTILE = 0.99  # the duration quantile of (38)


def _rules(params: GeneratorParams, block: int) -> dict:
    """The keywords that give ``targets.type_weights`` the params' target rules for ``block`` (None on `tiny`)."""
    return {"targeting": targeting_rules(params), "block": block}


def derived_adjacency(inst: Instance) -> tuple[tuple[str, str, float], ...]:
    """A_mm' = 1 when an edge joins a node in m and one in m' != m (either direction), symmetric (§2.4 profile)."""
    pairs = set()
    for e in inst.edges:
        a, b = inst.nodes[e.tail].region, inst.nodes[e.head].region
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    out = []
    for a, b in sorted(pairs):
        out.append((inst.regions[a], inst.regions[b], 1.0))
        out.append((inst.regions[b], inst.regions[a], 1.0))
    return tuple(out)


def active_regions(inst: Instance) -> tuple[str, ...]:
    """Regions hosting a node, in region order."""
    hosts = {n.region for n in inst.nodes}
    return tuple(r for i, r in enumerate(inst.regions) if i in hosts)


def chokepoint_regions(inst: Instance) -> frozenset[str]:
    """Regions adjacent to a chokepoint (Q58): those whose militarised baseline the closure calibration scales."""
    return frozenset(inst.regions[m] for c in inst.chokepoints for m in inst.chokepoint_adjacency.get(c, ()))


def stationary_conflict(P_yr) -> np.ndarray:
    """Stationary law of the weekly conflict chain Pi^c (28)."""
    Pi = weekly_conflict_matrix(P_yr)
    w, v = np.linalg.eig(Pi.T)
    s = np.real(v[:, np.argmin(abs(w - 1.0))])
    return s / s.sum()


def policy_multiplier(params: GeneratorParams) -> float:
    """E over pi^c of the policy multiplier (28): (1 - P(tension | z)) m_P(normal) + P(tension | z) m_P(tension).

    pi^c is the stationary law of the weekly conflict chain (``stationary_conflict``), the law the militarised
    multiplier is averaged over; P(tension | z) is ``regime.tension_stationary`` (1 - u_p at z = none).
    """
    pi_c = stationary_conflict(params.regime.P_yr)
    p_t = tension_stationary(np.arange(len(pi_c)), params.regime)
    m_normal, m_tension = params.hawkes.policy_by_tension
    return float(np.dot(pi_c, (1.0 - p_t) * m_normal + p_t * m_tension))


def stationary_baseline(inst: Instance, params: GeneratorParams, pi_c: np.ndarray | None = None) -> np.ndarray:
    """lambda-bar^0 (2R,) of (32): each region's baselines times the regime multipliers at their stationary means.

    The militarised multiplier E[m_M(z^c)] and the policy multiplier ``policy_multiplier``, both averaged over the
    stationary law pi^c of the weekly conflict chain (28) (``pi_c``, computed when not given); indexed [block * R +
    region], 0 for regions without a baseline. No baseline is checked here (``hawkes.baseline_vector`` does).
    """
    R = len(inst.regions)
    if pi_c is None:
        pi_c = stationary_conflict(params.regime.P_yr)
    mult_m = float(np.dot(pi_c, params.hawkes.militarised_by_conflict))
    mult_p = policy_multiplier(params)
    lam0 = np.zeros(2 * R)
    for r, bp, bm in params.hawkes.baselines:
        i = inst.region_index[r]
        lam0[P_BLOCK * R + i] = bp * mult_p
        lam0[M_BLOCK * R + i] = bm * mult_m
    return lam0


def closure_rate(inst: Instance, params: GeneratorParams) -> float:
    """Stationary militarised closures per week on all chokepoints, by (32) with mean regime multipliers.

    The militarised multiplier E[m_M(z^c)] and the policy multiplier ``policy_multiplier`` are both averaged over the
    stationary law pi^c of the weekly conflict chain (28) (``stationary_baseline``); the closure share of each region's
    type law is averaged over pi^c as well.
    """
    R = len(inst.regions)
    G = branching_matrix(inst, params)
    pi_c = stationary_conflict(params.regime.P_yr)
    Lam = stationary_rates(G, stationary_baseline(inst, params, pi_c))
    adjacency = adjacency_dict(inst, params)
    total = 0.0
    for m in range(R):
        share = 0.0
        for z, pz in enumerate(pi_c):  # the type law varies with the conflict state (regional conflicts need war)
            weights = dict(type_weights(inst, params.types.militarised, m, z, adjacency, **_rules(params, M_BLOCK)))
            share += pz * weights.get(MILITARISED_CLOSURE, 0.0)
        total += Lam[M_BLOCK * R + m] * share
    return float(total)


def tiny_table(inst: Instance) -> GeneratorParams:
    """The §2.4 profile table at the anchor rung before the closure calibration: uniform baselines on active regions."""
    act = active_regions(inst)
    per_region = ANCHOR_EVENTS_PER_YEAR / WEEKS_PER_YEAR * (1.0 - ANCHOR_GAMMA) / len(act)
    baselines = tuple((r, POLICY_SHARE * per_region, (1.0 - POLICY_SHARE) * per_region) for r in act)
    hawkes = HawkesParams(gamma=ANCHOR_GAMMA, trade_adjacency=derived_adjacency(inst), baselines=baselines)
    return GeneratorParams(profile="tiny", active_regions=act, hawkes=hawkes)


def closure_target() -> float:
    """The §2.4 calibration target in militarised closures per week on the chokepoints: 0.43 x 4/7 per year / 52."""
    return FLAGSHIP_CLOSURES_PER_YEAR * TINY_CHOKEPOINT_SHARE / WEEKS_PER_YEAR


def calibrate_closure_baselines(inst: Instance, params: GeneratorParams) -> tuple[tuple[str, float], ...]:
    """The (32) calibration of §2.4: (region, militarised baseline) of the chokepoint-adjacent regions, region order.

    Their militarised baselines are scaled by one factor so that ``closure_rate`` at ``params`` (the anchor rung) is
    ``closure_target()``; the rate is linear in those baselines (32), so the factor is exact up to rounding. On the
    packaged `tiny` the GULF baseline is 6.2358003e-4 per week (§2.4).

    Raises:
        ValueError: if the other regions alone already reach the target rate.

    """
    target = closure_target()
    adjacent = chokepoint_regions(inst)
    hawkes = params.hawkes
    rate0 = closure_rate(inst, params)
    zeroed = dataclasses.replace(
        hawkes, baselines=tuple((r, bp, 0.0 if r in adjacent else bm) for r, bp, bm in hawkes.baselines)
    )
    rate_other = closure_rate(inst, dataclasses.replace(params, hawkes=zeroed))
    scale = (target - rate_other) / (rate0 - rate_other)  # the closure rate is linear in those baselines (32)
    if scale <= 0:
        raise ValueError("closure calibration: other regions already exceed the target rate")
    return tuple((r, bm * scale) for r, _, bm in hawkes.baselines if r in adjacent)


def duration_quantiles(params: GeneratorParams, q: float = BURN_IN_QUANTILE) -> dict[str, float]:
    """F^{-1}_{T_q|ty}(q) in weeks per event type, T_q as (38) reads it (§4.3).

    Every type of ``params.laws.duration`` by its law (``Law.quantile``, mixtures by bisection); the regional conflict,
    a war-profile event, by its window |D_q| = max(T_q, w) + w of (37), (40), w = ``params.marks.war_profile_window``
    (``marks.war_windows`` at onset 0, the one rule of the window, SIMP-M2R2-12), monotone in T_q so its quantile is
    the window of F^{-1}(q); the port strike by the mixture of the stoppage and slowdown laws with weights
    ``stoppage_share`` and 1 - ``stoppage_share``.
    """
    out = {ty: law.quantile(q) for ty, law in params.laws.duration}
    if "regional_conflict" in out:
        w = float(params.marks.war_profile_window)
        out["regional_conflict"] = war_windows(0.0, out["regional_conflict"], w)[1]  # |D_q| of (40), one rule
    pois = params.poisson
    share = float(pois.stoppage_share)
    out["port_strike"] = mixture_quantile(q, ((share, pois.stoppage_duration), (1.0 - share, pois.slowdown_duration)))
    return out


def burn_in_bound(inst: Instance, params: GeneratorParams, rungs: tuple[float, ...] = RUNGS) -> float:
    """The right side of (38), maximised over ``rungs``: max_rung t_95 of (33) + max_ty F^{-1}_{T_q|ty}(0.99), weeks.

    t_95 = ln 20 / (beta (1 - spr(Gamma))) under the block Gamma of (31) at each rung, with the params' 1/beta; the
    quantiles are ``duration_quantiles``.

    Raises:
        ValueError: if spr(Gamma) >= 1 at a rung (no finite t_95).

    """
    t95 = 0.0
    for g in rungs:
        G = branching_matrix(inst, params.with_rung(g))
        if not spectral_radius(G) < 1.0:
            raise ValueError(f"(38): spr(Gamma) at rung {g} is {spectral_radius(G)}, so t_95 of (33) is not finite")
        t95 = max(t95, relaxation_time(G, params.hawkes.beta_inv))
    return t95 + max(duration_quantiles(params).values())


def check_burn_in(inst: Instance, params: GeneratorParams, rungs: tuple[float, ...] = RUNGS) -> None:
    """(38): B_burn of ``params`` covers ``burn_in_bound`` over the family's rungs.

    Raises:
        ValueError: if ``params.burn_in`` is shorter than the bound, naming t_95, the longest-tailed type and the bound.

    """
    bound = burn_in_bound(inst, params, rungs)
    if not params.burn_in >= bound:
        ty, qmax = max(duration_quantiles(params).items(), key=lambda kv: kv[1])
        raise ValueError(
            f"(38): B_burn {params.burn_in} weeks is below t_95 {bound - qmax:.1f} + F^-1_T(0.99) {qmax:.1f} ({ty})"
            f" = {bound:.1f} weeks over the rungs {rungs}"
        )


def tiny_profile(inst: Instance, gamma: float = ANCHOR_GAMMA) -> GeneratorParams:
    """The `tiny` generator profile of design §2.4 at policy-block rung ``gamma`` (baselines fixed at the anchor).

    The table of ``tiny_table`` with the militarised baselines of the chokepoint-adjacent regions set by the (32)
    calibration ``calibrate_closure_baselines``, run here on the table at the anchor rung. (38) is checked over the
    rungs (``check_burn_in``).

    Raises:
        ValueError: if the calibration fails or the burn-in misses (38).

    """
    params = tiny_table(inst)
    calibrated = dict(calibrate_closure_baselines(inst, params))
    baselines = tuple((r, bp, calibrated.get(r, bm)) for r, bp, bm in params.hawkes.baselines)
    params = dataclasses.replace(params, hawkes=dataclasses.replace(params.hawkes, baselines=baselines))
    check_burn_in(inst, params)
    return params.with_rung(gamma)


# ----- M5 shared interfaces: the `small` and `full` profile, Q97's burn-in rule, the ladder's rungs ------------------
# Written in M5's "Specify and interfaces" step (docs/099-m5-streams.md); every stub raises
# ``NotImplementedError("M5 stream <name>")``, naming the stream that fills it. Owner-level choices the stubs leave open
# are in docs/owner-queue.md (2026-09-28, M5); a stream builds the recommended default meanwhile.
FLAGSHIP_KINDS = ("small", "full")  # the instance kinds ``flagship_profile`` serves (§2.2; Q83)
# the sanction and export-control duration law of `small` and `full`, which material outages reuse: the Weibull fitted
# on TIES 4.0 (619 spells, 501 ended, 118 censored), DERIVED(TIES 4.0) (Q106, Q107):
# uv run --no-project --with numpy==2.4.5 --with scipy==1.18.1 --with xlrd==2.0.2 python
#     scripts/python/evidence/m5_sanction_law.py  -> docs/evidence/m5_sanction_law.txt (sample A, "Weibull")
TIES_WEIBULL_SHAPE, TIES_WEIBULL_SCALE_DAYS = TIES_WEIBULL.params  # 0.6590, 1523.17 (the one place: params.py)
STOCK_LOST_SHARE = 0.01  # Q97: the burn-in loses at most 1 % of the stationary active stock of a type
# the knobs of the difficulty ladder (§2.6 table; Q107's regime-length rung), in the table's order; "k_c" and
# "alpha_sp" change the instance, the others the generator parameters
KNOBS = (
    "gamma",
    "closure_rate",
    "severity",
    "beta_inv",
    "k_c",
    "alpha_sp",
    "pi_long_mil",
    "persistent_median",
    "n_cl",
    "tomlin",
    "sanction_regime",
)
# the knobs M5 plays and gates, γ by (73) (Q114's addendum); every other knob's rungs, the reported ones of §2.6
# included, are deferred to phase 4 (the owner's answer of 2026-09-29 to the M5 gate, Q114's addendum on the other
# knobs)
GATED_KNOBS = ("gamma",)


@dataclass(frozen=True)
class Rung:
    """One rung of the difficulty ladder of §2.6: a knob of ``KNOBS`` at one value, every other knob at the anchor.

    The γ rungs are the values of ``RUNGS``; another knob's rung values are the owner's (§2.6; §11 rows 18, 19, 55,
    67; Q107). All rungs of one size share one spawn key per episode through the monotone coupling and one burn-in
    (§2.6; Q87), so a rung never changes ``GeneratorParams.burn_in``.

    Raises:
        ValueError: on a knob outside ``KNOBS``, a non-finite or boolean value, or a γ value outside ``RUNGS``.

    """

    knob: str
    value: float

    def __post_init__(self) -> None:
        if self.knob not in KNOBS:
            raise ValueError(f"rung knob must be one of {KNOBS}, got {self.knob!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError(f"rung value must be a finite number, got {self.value!r}")
        if self.knob == "gamma" and self.value not in RUNGS:
            raise ValueError(f"a gamma rung is one of {RUNGS} (§2.6), got {self.value!r}")


ANCHOR_RUNG = Rung("gamma", ANCHOR_GAMMA)  # the anchor of the harm strata and of every ladder test (§7.4; Q88)


def apply_rung(inst: Instance, params: GeneratorParams, rung: Rung) -> tuple[Instance, GeneratorParams]:
    """(instance, generator parameters) of ``rung``: the anchor's with one knob changed (§2.6).

    γ is ``params.with_rung`` (baselines and burn-in fixed, Q39, Q87) on the instance at that rung, whose warm start is
    naive's steady state with that rung's F_Q (``Instance.at_rung``; §2.3, owner queue M5-O37 (b)); an instance with
    one start for every rung (`tiny`) is returned as it is.

    Raises:
        NotImplementedError: for any knob but γ (M5 plays γ only; the other knobs' rungs, the reported ones of §2.6
            included, are deferred to phase 4 by the owner's answer of 2026-09-29, Q114's addendum on the other knobs).

    """
    if rung.knob == "gamma":
        return inst.at_rung(rung.value), params.with_rung(rung.value)
    raise NotImplementedError(f"knob {rung.knob!r} has no rung rule: its rungs are deferred to phase 4 (§2.6; Q114)")


def rung_code(rung: Rung) -> int:
    """The rung component of the stream-22 key (§4.1): ``RUNGS.index`` for γ, so `tiny`'s keys never move.

    Raises:
        NotImplementedError: for any knob but γ (codes >= len(RUNGS), with the knob's rung table, in phase 4: Q114).

    """
    if rung.knob == "gamma":
        return RUNGS.index(rung.value)
    raise NotImplementedError(f"knob {rung.knob!r} has no stream-22 rung code: its rungs are deferred to phase 4, Q114")


# ----- M5 stream "generator": the `small` and `full` profile ----------------------------------------------------------
# Owner answers (2026-09-28: M5-O7 (a), M5-O8 (b)) and the owner queue's recommended defaults built until answered, each
# in one named place so an answer is a one-line change (docs/owner-queue.md, 2026-09-28, M5):
FLAGSHIP_RATE_REGIONS = 14  # M5-O12 (a): 12/yr over `full`'s 14 active regions; per-region rates held on `small` too
BURN_IN_TYPES: frozenset[str] | None = None  # M5-O9 (a): Q97's 1 % rule covers every type (None), one B per family
FLAGSHIP_DYADS = (("CN", "TW"),)  # M5-O14 (a): CN-TW the only dyad (Q59)
CLOSURE_SPLIT = "one_scale"  # M5-O8 (b), owner: the adjacency edit (instance data) with `tiny`'s one-scale (32) rule
# Q114 (owner 2026-09-29, "V3: fix both (Recommended)"): the sanction-rules round's V3 on `small` and `full`
# (docs/099-m5-sanction-rules.md, option 1; design §12 row "Sanction target rules V3 (Q114)")
FLAGSHIP_TARGETING: TargetingRules | None = TargetingRules(sanction_no_transit=True, policy_no_op=True)
# M5-O7 (a), owner: Gamma^MP = params.FLAGSHIP_CROSS_MP (0.01); M5-O13 (a): params.FlagshipMarkLaws.strait_closure_*
GAMMA_RUNGS = tuple(Rung("gamma", g) for g in RUNGS)  # the gated family of M5 (GATED_KNOBS), anchor first


class _WarWindow:
    """The storage window |D_q| = max(T_q, w) + w of a war-profile event, (37) and (40), as a duration law of Q97.

    mean = 2 w + int_w^inf S_T; the tail integral at x, with y = x - w, is w + int_w^inf S_T - y for y <= w (the window
    is at least 2 w) and int_y^inf S_T beyond (``Law.tail_integral`` of T).
    """

    def __init__(self, law: Law, window: float) -> None:
        self.law, self.window = law, float(window)

    def mean(self) -> float:
        return 2.0 * self.window + self.law.tail_integral(self.window)

    def tail_integral(self, x: float) -> float:
        y = x - self.window
        if y <= self.window:
            return self.window + self.law.tail_integral(self.window) - y
        return self.law.tail_integral(y)


class _Mixture:
    """A mixture of duration laws ((weight, law), ...) as a duration law of Q97: weighted means and tail integrals."""

    def __init__(self, parts) -> None:
        self.parts = tuple((float(w), law) for w, law in parts if w > 0.0)

    def mean(self) -> float:
        return math.fsum(w * law.mean() for w, law in self.parts)

    def tail_integral(self, x: float) -> float:
        return math.fsum(w * law.tail_integral(x) for w, law in self.parts)


def duration_laws(params: GeneratorParams) -> dict[str, object]:
    """Every event type's duration law as Q97 reads it (the types of ``duration_quantiles``, read the same way).

    Each type of ``params.laws.duration`` by its ``Law``; the regional conflict by its war-profile window |D_q| =
    max(T_q, w) + w of (37), (40) (w = ``params.marks.war_profile_window``); the port strike by the stoppage and
    slowdown mixture at ``stoppage_share``. Each value has ``mean()`` and ``tail_integral(x)`` in weeks.
    """
    out: dict[str, object] = dict(params.laws.duration)
    if "regional_conflict" in out:
        out["regional_conflict"] = _WarWindow(out["regional_conflict"], params.marks.war_profile_window)
    pois = params.poisson
    share = float(pois.stoppage_share)
    out["port_strike"] = _Mixture(((share, pois.stoppage_duration), (1.0 - share, pois.slowdown_duration)))
    return out


def stock_lost(law: Law, window_weeks: float) -> float:
    """The share of a type's stationary active stock a burn-in of ``window_weeks`` misses (Q97; §12 M5 row "Burn-in").

    int_w^inf S(x) dx / int_0^inf S(x) dx for the duration law's survival S (weeks), w = ``window_weeks``: the
    events of a stationary stream started before -w that are still active at week 0 (the equilibrium residual tail),
    ``law.tail_integral(w) / law.mean()`` in closed form (``laws.Law``; ``duration_laws`` for the war window and the
    strike mixture). Sample A's Weibull loses 0.0543 at w = 1,680 (docs/evidence/m5_sanction_law.txt, "lost(1680)").

    Raises:
        ValueError: if the window is negative or NaN, or the law's mean is not positive and finite.

    """
    if not window_weeks >= 0.0:
        raise ValueError(f"a burn-in window is >= 0 weeks, got {window_weeks!r}")
    mean = law.mean()
    if not (math.isfinite(mean) and mean > 0.0):
        raise ValueError(f"Q97's share needs a positive, finite mean duration, got {mean!r}")
    return law.tail_integral(window_weeks) / mean


def run_up(law: Law, share: float = STOCK_LOST_SHARE) -> float:
    """The smallest window w (weeks) with ``stock_lost(law, w) <= share`` (Q97's 1 % rule).

    ``stock_lost`` falls continuously in w (strictly while S > 0), so the window is its root, by Brent's method on a
    bracket doubled from max(E[T], 1) (relative tolerance 1e-12). Sample A's Weibull needs 3,059 weeks
    (docs/evidence/m5_sanction_law.txt, "B 1%").

    Raises:
        ValueError: if ``share`` is not in (0, 1), or no bracket up to 1e12 weeks reaches it.

    """
    if not 0.0 < share < 1.0:
        raise ValueError(f"Q97's share lies in (0, 1), got {share!r}")

    def gap(w: float) -> float:
        return stock_lost(law, w) - share

    if gap(0.0) <= 0.0:
        return 0.0
    hi = max(law.mean(), 1.0)
    while gap(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError(f"Q97: no window up to 1e12 weeks loses at most {share} of the stock")
    return float(optimize.brentq(gap, 0.0, hi, xtol=1e-9, rtol=1e-12))


def duration_run_ups(
    params: GeneratorParams, share: float = STOCK_LOST_SHARE, types: Collection[str] | None = None
) -> dict[str, float]:
    """``run_up`` of each type of ``duration_laws`` (``types`` None: every type; else those named), weeks.

    Raises:
        ValueError: if ``types`` names a type the generator has no duration law for.

    """
    laws = duration_laws(params)
    if types is not None:
        unknown = set(types) - set(laws)
        if unknown:
            raise ValueError(f"Q97: no duration law for {sorted(unknown)}; the types are {sorted(laws)}")
        laws = {ty: law for ty, law in laws.items() if ty in types}
    return {ty: run_up(law, share) for ty, law in laws.items()}


def burn_in_family(
    inst: Instance,
    params: GeneratorParams,
    rungs: tuple[Rung, ...],
    *,
    share: float = STOCK_LOST_SHARE,
    types: Collection[str] | None = None,
) -> float:
    """The burn-in of a `small` or `full` generator family (38) under Q97's rule, weeks.

    max over ``rungs`` of t_95 of (33) (``apply_rung`` of each; the block Gamma of (31) at its 1/beta), plus the largest
    ``run_up`` over the event types ``types`` (None: every type the generator draws, the regional conflict by its
    war-profile window and the port strike by its mixture, as ``duration_quantiles`` reads them; ``duration_run_ups``).
    One B for the family keeps the §2.6 coupling (Q87); whether Q97's rule covers every type or the long-lived ones
    only is the owner's (M5-O9; ``BURN_IN_TYPES`` holds the default, every type).

    Raises:
        ValueError: with no rung, if spr(Gamma) >= 1 at a rung (no finite t_95), or as ``duration_run_ups``.
        NotImplementedError: for a rung ``apply_rung`` has no rule for.

    """
    if not rungs:
        raise ValueError("a generator family has at least one rung")
    t95 = 0.0
    for rung in rungs:
        r_inst, p = apply_rung(inst, params, rung)
        G = branching_matrix(r_inst, p)
        if not spectral_radius(G) < 1.0:
            raise ValueError(f"(33): spr(Gamma) at {rung} is {spectral_radius(G)}, so t_95 is not finite")
        t95 = max(t95, relaxation_time(G, p.hawkes.beta_inv))
    return t95 + max(duration_run_ups(params, share, types).values())


def region_conflict_laws(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """(R, 3) the stationary law of each region's effective conflict state z^c (28) (Q59; §12 M5 generator rows).

    A region's own chain and every dyad chain run the weekly chain Pi^c of (28) (``stationary_conflict``, pi^c), taken
    independent at their stationary laws; a dyad at war sets both its regions to war (``regime.effective_conflict``), so
    a region in n dyads is at war with probability 1 - (1 - pi_war)^n (1 - pi_war) and in state z < war with pi_z
    (1 - pi_war)^n; any other region follows pi^c. The latent tilt g(X)/c_X has mean 1 (29) and is left out, as in
    `tiny`'s calibration (§2.4).

    Raises:
        ValueError: if a region class multiplier differs from 1 (its onset hazard, and so its law, would differ).

    """
    reg = params.regime
    if any(m != 1.0 for _, m in reg.class_multiplier):
        raise ValueError("region_conflict_laws reads one region class of multiplier 1 (§2.4 'Region classes')")
    pi = stationary_conflict(reg.P_yr)
    R = len(inst.regions)
    n_dyads = np.zeros(R, dtype=np.int64)
    for a, b in reg.dyads:
        n_dyads[inst.region_index[a]] += 1
        n_dyads[inst.region_index[b]] += 1
    out = np.tile(pi, (R, 1))
    for m in np.flatnonzero(n_dyads):
        calm = (1.0 - pi[WAR]) ** int(n_dyads[m])
        row = pi * calm
        row[WAR] = 1.0 - calm * (1.0 - pi[WAR])
        out[m] = row
    return out


def _flagship_rates(inst: Instance, params: GeneratorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Lambda (2R,) of (32), the effective conflict laws (R, 3), pi^c): the stationary rates at mean multipliers."""
    R = len(inst.regions)
    laws = region_conflict_laws(inst, params)
    p_t = tension_stationary(np.arange(laws.shape[1]), params.regime)  # P(tension | effective z^c) (28)
    m_normal, m_tension = params.hawkes.policy_by_tension
    mult_p = laws @ ((1.0 - p_t) * m_normal + p_t * m_tension)
    mult_m = laws @ np.asarray(params.hawkes.militarised_by_conflict, dtype=np.float64)
    lam0 = np.zeros(2 * R)
    for r, bp, bm in params.hawkes.baselines:
        i = inst.region_index[r]
        lam0[P_BLOCK * R + i] = bp * mult_p[i]
        lam0[M_BLOCK * R + i] = bm * mult_m[i]
    return stationary_rates(branching_matrix(inst, params), lam0), laws, stationary_conflict(params.regime.P_yr)


def flagship_closure_rates(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """Stationary militarised closures per week per chokepoint ordinal on `small` and `full`, by (32).

    `tiny`'s rule (``closure_rate``) with each region's effective conflict law (``region_conflict_laws``, so a dyad's
    wars raise its regions' multipliers and move their type laws, Q59), plus the dyad strait closures (M5-O13):

    - Lambda = (Id - Gamma)^{-1} lambda-bar^0 (32), each region's baselines times its mean multipliers, militarised
      E[m_M(z^c)] and policy E[(1 - P(tension | z^c)) m_P(normal) + P(tension | z^c) m_P(tension)] (28) under its
      effective law;
    - region m's militarised closures: Lambda_{M,m} times the closure share of its militarised type law averaged over
      its effective law, split evenly over its adjacent chokepoints (the closure target rule of §2.4);
    - a strait closure of chokepoint c for the pair (a, b): for m = a, b with partner p the other, Lambda_{M,m} times
      the regional-conflict share at war times P(dyad (m, p) at war) + P(m's own war, no dyad war) A_mp / sum A_m.,
      the counterpart rule of §4.4 (a dyad's partner at war takes precedence, else a region by A).

    Raises:
        ValueError: as ``region_conflict_laws`` and ``strait_chokepoints``, or if a strait region is in more than one
            dyad (its counterpart law is not written for that).

    """
    R = len(inst.regions)
    Lam, laws, pi = _flagship_rates(inst, params)
    adjacency = adjacency_dict(inst, params)
    mil = params.types.militarised
    ordinal = {c: i for i, c in enumerate(inst.chokepoints)}
    rates = np.zeros(len(inst.chokepoints))
    for m in range(R):
        lam_m = Lam[M_BLOCK * R + m]
        adjacent = [ordinal[c] for c in inst.chokepoints if m in inst.chokepoint_adjacency.get(c, ())]
        if lam_m == 0.0 or not adjacent:
            continue
        share = math.fsum(
            laws[m, z]
            * dict(type_weights(inst, mil, m, z, adjacency, **_rules(params, M_BLOCK))).get(MILITARISED_CLOSURE, 0.0)
            for z in range(laws.shape[1])
        )
        for i in adjacent:
            rates[i] += lam_m * share / len(adjacent)
    return rates + _strait_rates(inst, params, Lam, laws, pi, adjacency)


def _strait_rates(inst: Instance, params: GeneratorParams, Lam, laws, pi, adjacency) -> np.ndarray:
    """The dyad strait closures per week per chokepoint ordinal (M5-O13; ``flagship_closure_rates``' last item)."""
    R = len(inst.regions)
    mil = params.types.militarised
    ordinal = {c: i for i, c in enumerate(inst.chokepoints)}
    rates = np.zeros(len(inst.chokepoints))
    dyads = {frozenset((inst.region_index[a], inst.region_index[b])) for a, b in params.regime.dyads}
    for pair, c in strait_chokepoints(inst, params).items():
        for m in pair:
            (partner,) = pair - {m}
            if sum(m in d for d in dyads) > 1:
                raise ValueError(f"strait closure: region {inst.regions[m]} is in more than one dyad")
            p_dyad = float(pi[WAR]) if pair in dyads else 0.0
            p_own = float(laws[m, WAR]) - p_dyad  # own war with no dyad at war
            row = adjacency.get(m, {})
            total_a = math.fsum(w for w in row.values() if w > 0)
            a_share = row.get(partner, 0.0) / total_a if total_a > 0 else 0.0
            law = dict(type_weights(inst, mil, m, WAR, adjacency, **_rules(params, M_BLOCK)))
            w_rc = law.get(REGIONAL_CONFLICT, 0.0)
            rates[ordinal[c]] += Lam[M_BLOCK * R + m] * w_rc * (p_dyad + p_own * a_share)
    return rates


STRAIT_CLOSURE = "strait_closure"  # the key of the strait closures in ``flagship_type_rates`` (their own duration)


def flagship_type_rates(inst: Instance, params: GeneratorParams) -> dict[str, float]:
    """Stationary onsets per week per event type on `small` and `full`, by the flagship (32) rule; V17's Lambda_ty.

    Per block and region, Lambda_{B,m} (``flagship_closure_rates``' Lambda, effective conflict laws) times the type's
    share of the block's type law averaged over the region's effective law (a block with no feasible type makes no-op
    events, which omega never stores, and so does the policy block's no-op part under the params' ``targeting``, V2 of
    Q114: it is skipped); weather closures at the yearly rate / 52 per chokepoint and
    port strikes per struck region (active regions whose ports have a sea edge, ``events``' rule); the dyad strait
    closures under ``STRAIT_CLOSURE``, apart from the cluster's militarised closures, since their window max(T_q, 52)
    is their own duration. Keys are ``codes.EVENT_TYPES`` names plus ``STRAIT_CLOSURE``; the militarised closures plus
    the strait closures sum to ``flagship_closure_rates``.

    Raises:
        ValueError: as ``flagship_closure_rates``.

    """
    R = len(inst.regions)
    Lam, laws, pi = _flagship_rates(inst, params)
    adjacency = adjacency_dict(inst, params)
    out = dict.fromkeys(codes.EVENT_TYPES, 0.0)
    for block, shares in ((P_BLOCK, params.types.policy), (M_BLOCK, params.types.militarised)):
        for m in range(R):
            lam = Lam[block * R + m]
            if lam == 0.0:
                continue
            for z in range(laws.shape[1]):
                for code, w in type_weights(inst, shares, m, z, adjacency, **_rules(params, block)):
                    if code == NOOP:  # V2's no-op part (Q114): omega never stores it, so it has no onset rate
                        continue
                    out[codes.EVENT_TYPES[code]] += lam * laws[m, z] * w
    pois = params.poisson
    out["weather_closure"] = pois.weather_per_chokepoint_year / WEEKS_PER_YEAR * len(inst.chokepoints)
    out["port_strike"] = pois.strike_per_region_year / WEEKS_PER_YEAR * len(strike_regions(inst, params))
    out[STRAIT_CLOSURE] = float(_strait_rates(inst, params, Lam, laws, pi, adjacency).sum())
    return out


def calibrate_flagship_closures(inst: Instance, params: GeneratorParams) -> tuple[tuple[str, float], ...]:
    """The one-scale (32) calibration of `small` and `full` (M5-O8 (b)): (region, militarised baseline), region order.

    The militarised baselines of the chokepoint-adjacent regions (``chokepoint_regions``, on the instance's adjacency)
    are scaled by one factor so that ``flagship_closure_rates`` at ``params`` (the anchor rung) sums to 0.43 / 52 per
    week on the seven chokepoints (Q52, Q76 D5); the sum is linear in those baselines (32), so the factor is exact up to
    rounding. `tiny`'s rule (``calibrate_closure_baselines``) with the target unscaled and the flagship rates.

    Raises:
        ValueError: if the other regions alone already reach the target.
        NotImplementedError: if ``CLOSURE_SPLIT`` names a rule other than "one_scale".

    """
    if CLOSURE_SPLIT != "one_scale":
        raise NotImplementedError(f"closure split {CLOSURE_SPLIT!r} (M5-O8) has no rule yet")
    target = FLAGSHIP_CLOSURES_PER_YEAR / WEEKS_PER_YEAR
    adjacent = chokepoint_regions(inst)
    hawkes = params.hawkes
    rate1 = float(flagship_closure_rates(inst, params).sum())
    zeroed = dataclasses.replace(
        hawkes, baselines=tuple((r, bp, 0.0 if r in adjacent else bm) for r, bp, bm in hawkes.baselines)
    )
    rate0 = float(flagship_closure_rates(inst, dataclasses.replace(params, hawkes=zeroed)).sum())
    scale = (target - rate0) / (rate1 - rate0)  # the closure rate is linear in those baselines (32)
    if not scale > 0.0:
        raise ValueError("closure calibration: other regions already exceed the target rate")
    return tuple((r, bm * scale) for r, _, bm in hawkes.baselines if r in adjacent)


def closure_targets(inst: Instance) -> Mapping[str, float]:
    """Militarised closures per week per chokepoint id at the anchor rung, the targets of the (32) calibration.

    They sum to 0.43 / 52 on `small` and `full`, which carry all seven flagship chokepoints (§2.2; Q52, Q76). Under the
    owner's M5-O8 answer (b) the split is not an input: it is what the one-scale calibration gives on the instance's
    chokepoint adjacency (``flagship_closure_rates`` of the calibrated anchor profile), which V22 reports; option (c)
    would make these the inputs.

    Raises:
        ValueError: on an instance kind other than `small` or `full`.

    """
    rates = flagship_closure_rates(inst, flagship_profile(inst))
    return {inst.nodes[c].id: float(rates[i]) for i, c in enumerate(inst.chokepoints)}


def flagship_table(inst: Instance) -> GeneratorParams:
    """The `small` and `full` profile table at the anchor rung before the closure calibration (§12 M5 generator rows).

    The counterpart of ``tiny_table``: the active regions; the derived adjacency (§2.4 rule); per active region
    12/52 (1 - 0.62) / 14 events per week split policy 0.70 / militarised 0.30 (M5-O12 (a)); the dyads of
    ``FLAGSHIP_DYADS``; Gamma^MP 0.01 (M5-O7 (a), ``FlagshipHawkesParams``); the TIES Weibull for sanctions and
    material outages (Q106) and the strait closures (``FlagshipMarkLaws``); Q114's ``TargetingRules`` (V3: no
    sanction on a transit leg, the policy block's infeasible share a no-op part; design §12 row "Sanction target rules
    V3 (Q114)"; `tiny` leaves the field None); and
    ``burn_in_family``'s B over the γ rungs, rounded up to a whole week. Each value carries its provenance
    (``params.FlagshipGeneratorParams`` and the subclasses), so ``params.provenance`` never reports `tiny`'s strings
    for a value that differs.

    Raises:
        ValueError: on an instance kind other than `small` or `full`, a dyad or strait region the instance lacks, or a
            strait chokepoint it lacks.

    """
    if inst.kind not in FLAGSHIP_KINDS:
        raise ValueError(f"the flagship profile serves {FLAGSHIP_KINDS}, not kind {inst.kind!r}")
    for a, b in FLAGSHIP_DYADS:
        if a not in inst.region_index or b not in inst.region_index:
            raise ValueError(f"dyad ({a}, {b}): a region the instance lacks")
    act = active_regions(inst)
    per_region = ANCHOR_EVENTS_PER_YEAR / WEEKS_PER_YEAR * (1.0 - ANCHOR_GAMMA) / FLAGSHIP_RATE_REGIONS
    baselines = tuple((r, POLICY_SHARE * per_region, (1.0 - POLICY_SHARE) * per_region) for r in act)
    hawkes = FlagshipHawkesParams(gamma=ANCHOR_GAMMA, trade_adjacency=derived_adjacency(inst), baselines=baselines)
    params = FlagshipGeneratorParams(
        profile=inst.kind,
        active_regions=act,
        regime=RegimeParams(dyads=FLAGSHIP_DYADS),
        hawkes=hawkes,
        laws=FlagshipMarkLaws(),
        targeting=FLAGSHIP_TARGETING,
    )
    strait_chokepoints(inst, params)  # the named regions and chokepoints exist
    burn_in = math.ceil(burn_in_family(inst, params, GAMMA_RUNGS, types=BURN_IN_TYPES))
    return dataclasses.replace(params, burn_in=burn_in)


def _flagship_anchor(inst: Instance) -> GeneratorParams:
    params = flagship_table(inst)
    calibrated = dict(calibrate_flagship_closures(inst, params))
    baselines = tuple((r, bp, calibrated.get(r, bm)) for r, bp, bm in params.hawkes.baselines)
    params = dataclasses.replace(params, hawkes=dataclasses.replace(params.hawkes, baselines=baselines))
    check_burn_in(inst, params)  # (38) holds too: Q97's run-up exceeds every type's p99
    return params


def flagship_profile(inst: Instance, gamma: float = ANCHOR_GAMMA) -> GeneratorParams:
    """The generator profile of `small` and `full` at policy-block rung ``gamma`` (baselines and B fixed at the anchor).

    ``flagship_table`` with the militarised baselines of the chokepoint-adjacent regions set by the one-scale (32)
    calibration (``calibrate_flagship_closures``), run on the table at the anchor rung; (38) is checked as well
    (``check_burn_in``). Built once per instance and kept in ``Instance.memo`` (deterministic on one machine; another
    machine's LAPACK may round the calibration's last bits differently, which Q94 accepts), then moved to the rung.

    Raises:
        ValueError: as ``flagship_table``, if the calibration fails, or if the burn-in misses (38).

    """
    return inst.memo("disruption.profiles.flagship_anchor", _flagship_anchor).with_rung(gamma)
