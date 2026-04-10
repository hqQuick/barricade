from __future__ import annotations

import statistics
from collections import Counter

from .constants import (
    COOLING_STABILITY_THRESHOLD,
    COOLING_SUCCESS_THRESHOLD,
    EXPLOIT_ENTRY_THRESHOLD,
    EXPLOIT_EXIT_THRESHOLD,
    EXPLOIT_HOLD,
    EXPLOIT_RETENTION_BONUS,
    REGIME_CENTER,
)
from .models import ControllerState


__all__ = [
    "boost_for_missing_niche",
    "decide_controller",
    "exploit_dwell_stats",
    "exploit_entry_barrier",
    "exploit_score",
    "maybe_loopback",
    "anti_reinforcement_candidates",
    "motif_reinforcement_candidates",
    "niche_target_counts",
    "niche_overrepresentation_penalty",
    "phase_signal",
    "rotation_market_signal",
    "regime_from_temperature",
    "rolling_momentum",
    "temperature_to_episodes",
    "update_motif_market",
]


def rotation_market_signal(metrics):
    try:
        return max(
            0.0, min(1.0, float(metrics.get("rotation_market_signal", 0.0) or 0.0))
        )
    except (TypeError, ValueError):
        return 0.0


def _governance_variance_ratio(metrics):
    variance = max(0.0, float(metrics.get("governance_variance", 0.0) or 0.0))
    governed_mean = abs(float(metrics.get("governed_mean", 0.0) or 0.0))
    if governed_mean > 0.0:
        return variance / max(1.0, variance + governed_mean * governed_mean)
    return min(1.0, variance / 1600.0)


def _smoothed_rotation_market_signal(state, metrics, smoothing=0.35):
    current = rotation_market_signal(metrics)
    if not state.rotation_market_initialized:
        state.rotation_market_initialized = True
        state.rotation_market_ema = current
    else:
        state.rotation_market_ema = (
            smoothing * current + (1.0 - smoothing) * state.rotation_market_ema
        )
    return state.rotation_market_ema


PHASE_SIGNAL_BASE = 0.55


def _phase_signal_deficit(value, floor, weight):
    return max(0.0, floor - value) * weight


def _phase_signal_excess(value, ceiling, weight):
    return max(0.0, value - ceiling) * weight


def phase_signal(metrics):
    # These pressure terms are tuned against the benchmark telemetry. Positive
    # values heat the controller; negative values cool it.
    temp = PHASE_SIGNAL_BASE

    task_threshold_pressure = _phase_signal_deficit(
        metrics["task_threshold_pass_mean"], 0.74, 1.05
    ) - _phase_signal_excess(metrics["task_threshold_pass_mean"], 0.82, 0.28)
    specialization_pressure = _phase_signal_deficit(
        metrics["specialization_entropy"], 0.58, 0.55
    ) - _phase_signal_excess(metrics["specialization_entropy"], 0.60, 0.18)
    economic_stability_pressure = _phase_signal_deficit(
        metrics["economic_stability_mean"], 0.52, 0.50
    )
    governance_variance_pressure = min(0.22, 0.22 * _governance_variance_ratio(metrics))
    motif_pressure = -_phase_signal_excess(metrics["motif_entropy"], 0.78, 0.18)
    lineage_pressure = -_phase_signal_excess(metrics["lineage_entropy"], 0.82, 0.08)
    stability_pressure = -_phase_signal_excess(
        metrics["stability_score_mean"], 0.72, 0.20
    )
    selection_pressure = -_phase_signal_excess(
        float(metrics.get("selection_score_mean", 0.0) or 0.0), 0.60, 0.10
    )
    plan_axis_pressure = -_phase_signal_excess(
        metrics.get("optimizer_plan_axis_mean", 0.0), 0.55, 0.10
    )
    mixed_pressure = _phase_signal_excess(
        metrics.get("optimizer_mixed_pressure_mean", 0.0), 0.20, 0.12
    )
    ast_depth_pressure = _phase_signal_excess(
        metrics.get("optimizer_ast_depth_mean", 0.0), 1.0, 0.02
    )
    parallax_pressure = _phase_signal_excess(
        metrics.get("parallax_pressure", 0.0), 0.18, 0.10
    )
    rotation_market = rotation_market_signal(metrics)
    rotation_condition_pressure = _phase_signal_excess(
        metrics.get("rotation_condition", 0.0),
        1.5,
        0.01 + 0.01 * (1.0 - rotation_market),
    )
    orthogonality_pressure = _phase_signal_deficit(
        metrics.get("orthogonality_spread", 0.0), 0.20, 0.12
    )
    devolution_pressure = _phase_signal_excess(
        metrics.get("devolution_pressure", 0.0), 0.20, 0.04
    )
    rotation_market_pressure = _phase_signal_deficit(rotation_market, 0.45, 0.05)

    temp += task_threshold_pressure
    temp += specialization_pressure
    temp += economic_stability_pressure
    temp += governance_variance_pressure
    temp += motif_pressure
    temp += lineage_pressure
    temp += stability_pressure
    temp += selection_pressure
    temp += plan_axis_pressure
    temp += mixed_pressure
    temp += ast_depth_pressure
    temp += parallax_pressure
    temp += rotation_condition_pressure
    temp += orthogonality_pressure
    temp += devolution_pressure
    temp += rotation_market_pressure
    return max(0.0, min(1.0, temp))


def regime_from_temperature(temp):
    if temp < 0.52:
        return "exploit"
    if temp < 0.78:
        return "balanced"
    return "explore"


def temperature_to_episodes(temp, min_ep, max_ep):
    frac = 1.0 - temp
    return int(round(min_ep + frac * (max_ep - min_ep)))


def rolling_momentum(state):
    if len(state.recent_governed) < 2:
        return 0.0

    governed_start = state.recent_governed[0]
    governed_end = state.recent_governed[-1]
    threshold_start = state.recent_threshold[0]
    threshold_end = state.recent_threshold[-1]
    stability_start = state.recent_stability[0]
    stability_end = state.recent_stability[-1]

    governed_trend = (governed_end - governed_start) / max(
        40.0, abs(governed_start), 1.0
    )
    threshold_trend = threshold_end - threshold_start
    stability_trend = stability_end - stability_start

    momentum = 0.42 * governed_trend + 0.34 * threshold_trend + 0.24 * stability_trend
    return max(-1.0, min(1.0, momentum))


def exploit_score(metrics, state):
    momentum = max(0.0, rolling_momentum(state))
    entropy_term = (
        0.45 * metrics["motif_entropy"] + 0.55 * metrics["specialization_entropy"]
    )
    selection_term = max(
        0.0, min(1.0, float(metrics.get("selection_score_mean", 0.0) or 0.0))
    )
    variance_ratio = _governance_variance_ratio(metrics)
    stability_term = (
        0.6 * metrics["stability_score_mean"] + 0.4 * metrics["economic_stability_mean"]
    )
    performance_term = (
        0.7 * metrics["task_threshold_pass_mean"]
        + 0.3 * metrics["solve_rate_test_mean"]
    )
    variance_term = max(0.0, 1.0 - variance_ratio)
    geometry_term = (
        0.5 * metrics.get("optimizer_plan_axis_mean", 0.0)
        + 0.25 * metrics.get("optimizer_summary_axis_mean", 0.0)
        + 0.25 * metrics.get("optimizer_balance_mean", 0.0)
    )
    combinator_term = min(
        1.0, metrics.get("optimizer_discover_chain_mean", 0.0) / 3.0
    ) + 0.2 * metrics.get("optimizer_combinator_mass_mean", 0.0)
    orthogonality_term = min(1.0, metrics.get("orthogonality_spread", 0.0) / 2.0)
    return (
        0.36 * performance_term
        + 0.28 * stability_term
        + 0.14 * entropy_term
        + 0.16 * momentum
        + 0.06 * variance_term
        + 0.06 * geometry_term
        + 0.04 * combinator_term
        + 0.03 * orthogonality_term
        + 0.03 * selection_term
    )


def exploit_entry_barrier(metrics, state):
    barrier = EXPLOIT_ENTRY_THRESHOLD
    barrier -= min(0.07, max(0.0, rolling_momentum(state)) * 0.14)
    barrier -= min(0.06, max(0.0, metrics["task_threshold_pass_mean"] - 0.78) * 0.12)
    barrier -= min(0.05, max(0.0, metrics["stability_score_mean"] - 0.70) * 0.10)
    barrier -= min(0.05, max(0.0, metrics["specialization_entropy"] - 0.56) * 0.12)
    barrier -= min(
        0.03,
        max(0.0, float(metrics.get("selection_score_mean", 0.0) or 0.0) - 0.58) * 0.08,
    )
    barrier -= min(0.04, max(0.0, metrics.get("optimizer_plan_axis_mean", 0.0)) * 0.05)
    barrier += min(
        0.04, max(0.0, metrics.get("optimizer_mixed_pressure_mean", 0.0) - 0.18) * 0.08
    )
    barrier += min(0.04, max(0.0, metrics.get("rotation_condition", 0.0) - 1.5) * 0.01)
    barrier += 0.03 if state.phase_hold > 0 else 0.0
    return max(0.42, min(0.66, barrier))


def exploit_dwell_stats(controller_trace):
    runs = []
    current_trial = None
    current_len = 0

    for row in controller_trace:
        trial = row["trial"]
        if row["regime"] == "exploit":
            if current_trial == trial:
                current_len += 1
            else:
                if current_len:
                    runs.append(current_len)
                current_trial = trial
                current_len = 1
        else:
            if current_len:
                runs.append(current_len)
                current_len = 0
            current_trial = trial

    if current_len:
        runs.append(current_len)

    if not runs:
        return {"runs": [], "count": 0, "mean": 0.0, "max": 0, "total": 0}

    return {
        "runs": runs,
        "count": len(runs),
        "mean": round(statistics.mean(runs), 3),
        "max": max(runs),
        "total": sum(runs),
    }


def maybe_loopback(state, metrics, generation):
    triggered = False
    reasons = []
    if state.regime == "explore":
        shallow_gain = metrics["governed_mean"] - state.last_governed < 1.5
        bad_threshold = metrics["task_threshold_pass_mean"] < 0.68
        low_spec = metrics["specialization_entropy"] < 0.34
        if (bad_threshold and low_spec) or (low_spec and shallow_gain):
            state.stuck_counter += 1
        else:
            state.stuck_counter = max(0, state.stuck_counter - 1)
        if state.stuck_counter >= 3:
            state.temperature = 0.58
            state.regime = "balanced"
            state.phase_hold = 2
            state.stuck_counter = 0
            triggered = True
            reasons.append("loopback_escape_to_balanced")

    if state.regime == "exploit" and metrics.get("orthogonality_spread", 0.0) < 0.04:
        state.temperature = min(0.84, state.temperature + 0.08)
        state.regime = "balanced"
        state.phase_hold = max(state.phase_hold, 1)
        triggered = True
        reasons.append("loopback_restore_orthogonality")

    if state.regime == "balanced":
        if (
            metrics["profit_mean"] < -8
            and metrics["economic_stability_mean"] < 0.32
            and generation > 6
        ):
            state.temperature = min(0.88, state.temperature + 0.14)
            state.regime = "explore"
            state.phase_hold = 1
            triggered = True
            reasons.append("reheat_to_explore")

    return triggered, reasons


def decide_controller(
    state: ControllerState,
    metrics: dict[str, float | int],
    min_ep: int,
    max_ep: int,
    generation: int,
) -> tuple[ControllerState, int, bool, list[str]]:
    reasons = []
    state.recent_governed.append(metrics["governed_mean"])
    state.recent_threshold.append(metrics["task_threshold_pass_mean"])
    state.recent_stability.append(metrics["stability_score_mean"])

    if state.replication_hold > 0:
        state.replication_hold -= 1
        reasons.append("replication_hold")

    smoothed_rotation_market = _smoothed_rotation_market_signal(state, metrics)
    metrics = dict(metrics)
    metrics["rotation_market_signal"] = smoothed_rotation_market

    target_temp = phase_signal(metrics)
    state.temperature = 0.76 * state.temperature + 0.24 * target_temp

    if (
        metrics["task_threshold_pass_mean"] >= COOLING_SUCCESS_THRESHOLD
        and metrics["stability_score_mean"] >= COOLING_STABILITY_THRESHOLD
    ):
        state.temperature = max(0.16, state.temperature - 0.14)
        state.balanced_success_counter += 1
        reasons.append("cool_success")
    else:
        state.balanced_success_counter = max(0, state.balanced_success_counter - 1)

    variance_ratio = _governance_variance_ratio(metrics)

    if variance_ratio > 0.22:
        state.temperature = min(0.96, state.temperature + 0.06)
        reasons.append("heat_variance")
    if metrics["specialization_entropy"] < 0.28:
        state.temperature = min(0.95, state.temperature + 0.05)
        reasons.append("heat_specialization_collapse")
    if (
        metrics["economic_stability_mean"] > 0.68
        and metrics["task_threshold_pass_mean"] > 0.80
    ):
        state.temperature = max(0.20, state.temperature - 0.08)
        reasons.append("cool_economic_success")
    if metrics.get("parallax_pressure", 0.0) > 0.25:
        state.temperature = min(0.98, state.temperature + 0.05)
        reasons.append("heat_parallax_pressure")
    if metrics.get("rotation_condition", 0.0) > 2.0:
        rotation_market = rotation_market_signal(metrics)
        state.temperature = min(
            0.98, state.temperature + 0.02 + 0.02 * (1.0 - rotation_market)
        )
        reasons.append("heat_rotation_condition")
    if metrics.get("orthogonality_spread", 0.0) < 0.05:
        state.temperature = min(0.97, state.temperature + 0.04)
        reasons.append("heat_orthogonality_collapse")

    if state.phase_hold > 0:
        state.phase_hold -= 1
        reasons.append("phase_hold")
    else:
        state.regime = regime_from_temperature(state.temperature)

        current_exploit_score = exploit_score(metrics, state)
        current_barrier = exploit_entry_barrier(metrics, state)
        current_momentum = rolling_momentum(state)

        if state.regime in ("balanced", "explore"):
            if current_exploit_score >= current_barrier and current_momentum > 0.0:
                state.temperature = min(state.temperature, 0.36)
                state.regime = "exploit"
                state.phase_hold = EXPLOIT_HOLD
                state.exploit_stay_counter = 0
                state.exploit_exit_counter = 0
                reasons.append(f"enter_exploit_score_{current_exploit_score:.3f}")
            elif (
                current_exploit_score >= current_barrier - 0.03
                and current_momentum > 0.0
            ):
                state.exploit_stay_counter += 1
                reasons.append("exploit_capture_zone")
                if state.exploit_stay_counter >= 2:
                    state.temperature = min(state.temperature, 0.38)
                    state.regime = "exploit"
                    state.phase_hold = EXPLOIT_HOLD
                    state.exploit_stay_counter = 0
                    state.exploit_exit_counter = 0
                    reasons.append(f"enter_exploit_capture_{current_exploit_score:.3f}")
            elif (
                metrics["task_threshold_pass_mean"] > 0.78
                and metrics["economic_stability_mean"] > 0.62
                and metrics["specialization_entropy"] > 0.40
            ):
                state.temperature = max(0.48, min(0.68, state.temperature))
                state.regime = "balanced"
                reasons.append("stabilize_balanced")
            else:
                state.exploit_stay_counter = max(0, state.exploit_stay_counter - 1)

        if state.regime == "exploit":
            state.exploit_dwell_total += 1

            if current_exploit_score >= current_barrier and current_momentum > 0.0:
                state.temperature = max(
                    0.16, min(state.temperature, 0.34) - EXPLOIT_RETENTION_BONUS
                )
                state.phase_hold = max(state.phase_hold, 2)
                state.exploit_exit_counter = 0
                state.exploit_stay_counter += 1
                reasons.append("exploit_retention_bonus")
            else:
                state.exploit_stay_counter = max(0, state.exploit_stay_counter - 1)

            weak_exit = (
                current_exploit_score < EXPLOIT_EXIT_THRESHOLD
                or current_momentum < -0.02
                or metrics["economic_stability_mean"] < 0.45
            )
            if weak_exit:
                state.exploit_exit_counter += 1
            else:
                state.exploit_exit_counter = max(0, state.exploit_exit_counter - 1)

            if state.exploit_exit_counter >= 2:
                state.temperature = 0.58
                state.regime = "balanced"
                state.phase_hold = 1
                state.exploit_exit_counter = 0
                state.exploit_stay_counter = 0
                state.exploit_bounce_count += 1
                reasons.append(
                    f"exploit_exit_to_balanced_score_{current_exploit_score:.3f}"
                )
            elif variance_ratio > 0.16:
                state.temperature = 0.82
                state.regime = "explore"
                state.phase_hold = 1
                state.exploit_exit_counter = 0
                state.exploit_stay_counter = 0
                reasons.append("exploit_reheat")

    loopback_triggered, loop_reasons = maybe_loopback(state, metrics, generation)
    reasons.extend(loop_reasons)

    if state.phase_hold == 0:
        state.regime = regime_from_temperature(state.temperature)

    center = REGIME_CENTER[state.regime]
    state.temperature = 0.72 * state.temperature + 0.28 * center

    episodes = temperature_to_episodes(state.temperature, min_ep, max_ep)
    state.recent_regimes.append(state.regime)
    state.last_governed = metrics["governed_mean"]

    if not reasons:
        reasons.append("hold")

    return state, episodes, loopback_triggered, reasons


def update_motif_market(motif_market, pop):
    for x in pop:
        key = x.motif_sig
        if not key:
            continue
        slot = motif_market.setdefault(
            key,
            {
                "count": 0,
                "mean_governed": 0.0,
                "mean_profit": 0.0,
                "mean_stability": 0.0,
                "specializations": Counter(),
            },
        )
        slot["count"] += 1
        slot["mean_governed"] += (x.governed_fitness - slot["mean_governed"]) / slot[
            "count"
        ]
        slot["mean_profit"] += (x.profit - slot["mean_profit"]) / slot["count"]
        slot["mean_stability"] += (x.stability_score - slot["mean_stability"]) / slot[
            "count"
        ]
        slot["specializations"][x.specialization] += 1


def motif_reinforcement_candidates(motif_market, specialization):
    viable = []
    for motif, meta in motif_market.items():
        spec_hits = meta["specializations"].get(specialization, 0)
        score = (
            meta["mean_governed"] * 0.55
            + meta["mean_stability"] * 45
            + meta["mean_profit"] * 0.20
            + spec_hits * 1.5
        )
        if meta["count"] >= 3 and meta["mean_stability"] >= 0.68:
            viable.append((score, motif))
    viable.sort(reverse=True)
    return viable[:10]


def anti_reinforcement_candidates(motif_market, specialization):
    toxic = []
    for motif, meta in motif_market.items():
        spec_hits = meta["specializations"].get(specialization, 0)
        mean_governed = float(
            meta.get("mean_governed", meta.get("best_governed_fitness", 0.0)) or 0.0
        )
        mean_profit = float(meta.get("mean_profit", 0.0) or 0.0)
        mean_stability = float(meta.get("mean_stability", 0.0) or 0.0)
        score = (
            (1.0 - min(1.0, mean_stability)) * 45.0
            + max(0.0, -mean_profit) * 0.20
            + max(0.0, 0.45 - mean_governed / 100.0) * 20.0
            + spec_hits * 0.5
        )
        if meta["count"] >= 3 and mean_stability <= 0.35:
            toxic.append((score, motif))
    toxic.sort(reverse=True)
    return toxic[:10]


def niche_target_counts(population):
    summary = max(10, population // 7 + 1)
    patch = population // 4
    plan = population // 5
    summary_bridge = max(8, population // 9)
    return {
        "summary": summary,
        "patch": patch,
        "plan": plan,
        "summary_bridge": summary_bridge,
        "generalist": max(0, population - (summary + patch + plan + summary_bridge)),
    }


def boost_for_missing_niche(specialization, counts, targets):
    deficit = max(0, targets.get(specialization, 0) - counts.get(specialization, 0))
    return 1.0 + 0.15 * deficit


def niche_overrepresentation_penalty(specialization, counts, targets):
    target = max(1, targets.get(specialization, 0) or 0)
    current = max(0, counts.get(specialization, 0) or 0)
    excess = max(0, current - target)
    if excess <= 0:
        return 0.0
    coefficients = {
        "summary": 1.15,
        "summary_bridge": 0.50,
        "patch": 0.40,
        "plan": 0.35,
        "generalist": 0.25,
    }
    return min(5.0, coefficients.get(specialization, 0.25) * excess)
