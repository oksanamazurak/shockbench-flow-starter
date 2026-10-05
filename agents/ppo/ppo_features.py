"""Early-warning observation features shared by ``agents/mine_heuristic/train_ppo.py`` (training) and ``agent.py`` (inference).

Numpy + stdlib only, no import-time side effects, so the training script can import it before any ``policy.pt``
exists and the server can import it inside the ``agents/ppo`` submission.
"""

import numpy as np


def summarize_warnings(
    obs: dict, action_edges: np.ndarray, capacity: np.ndarray, week: float, horizon: int
) -> np.ndarray:
    """10 fixed-size early-warning features, from the padded lists the public example drops entirely.

    Order (weeks clipped to [0, horizon], sentinel ``horizon`` for "none pending" / "no known end"):
    pp_mine_count, pp_mine_nearest_weeks, pp_mine_capacity_fraction, pp_total_count, pp_total_nearest_weeks,
    closure_active_count, closure_nearest_end_weeks, messages_threat_count, messages_final_notice_count,
    messages_nearest_stated_effective_weeks.
    """

    def nearest_weeks(target_week: np.ndarray, observed: np.ndarray, selected: np.ndarray | None = None) -> float:
        live = np.asarray(observed, dtype=bool)
        if selected is not None:
            live = live & selected
        if not live.any():
            return float(horizon)
        return float(np.clip(np.asarray(target_week, dtype=np.float64)[live] - week, 0.0, horizon).min())

    pp_edge = np.asarray(obs["pending_prohibitions.edge"])
    pp_observed = np.asarray(obs["pending_prohibitions.effective_week.observed"], dtype=bool)
    pp_week = obs["pending_prohibitions.effective_week"]
    pp_mine = pp_observed & np.isin(pp_edge, action_edges)
    pp_mine_count = float(pp_mine.sum())
    pp_mine_nearest_weeks = nearest_weeks(pp_week, pp_observed, pp_mine)
    mine_capacity = float(capacity.sum())
    pp_mine_capacity_fraction = (
        float(capacity[np.isin(action_edges, pp_edge[pp_mine])].sum()) / mine_capacity if mine_capacity > 0 else 0.0
    )
    pp_total_count = float(pp_observed.sum())
    pp_total_nearest_weeks = nearest_weeks(pp_week, pp_observed)

    cl_observed = np.asarray(obs["closure_end.chokepoint.observed"], dtype=bool)
    cl_week = obs["closure_end.end_week"]
    cl_week_observed = np.asarray(obs["closure_end.end_week.observed"], dtype=bool)
    closure_active_count = float(cl_observed.sum())
    closure_nearest_end_weeks = nearest_weeks(cl_week, cl_week_observed)

    msg_kind = np.asarray(obs["messages.kind"])
    msg_observed = np.asarray(obs["messages.kind.observed"], dtype=bool)
    messages_threat_count = float((msg_observed & (msg_kind == 2)).sum())
    messages_final_notice_count = float((msg_observed & (msg_kind == 1)).sum())
    msg_eff_week = obs["messages.stated_effective_week"]
    msg_eff_observed = np.asarray(obs["messages.stated_effective_week.observed"], dtype=bool)
    messages_nearest_stated_effective_weeks = nearest_weeks(msg_eff_week, msg_eff_observed)

    return np.array(
        [
            pp_mine_count,
            pp_mine_nearest_weeks,
            pp_mine_capacity_fraction,
            pp_total_count,
            pp_total_nearest_weeks,
            closure_active_count,
            closure_nearest_end_weeks,
            messages_threat_count,
            messages_final_notice_count,
            messages_nearest_stated_effective_weeks,
        ],
        dtype=np.float64,
    )
