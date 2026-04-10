from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .constants import ALL_TOKENS, CHARTS

if TYPE_CHECKING:
    from .models import Individual


__all__ = [
    "adversarial_push",
    "build_protected_tokens",
    "crossover",
    "mutate",
    "random_individual",
    "random_token",
    "replicate_scout",
    "specialization_pair_bonus",
    "scout_teleport",
]


def random_token(rng: random.Random, macro_lib: Mapping[str, Sequence[str]]) -> str:
    return rng.choice(ALL_TOKENS + list(macro_lib.keys()))


def random_individual(
    rng: random.Random, macro_lib: Mapping[str, Sequence[str]], lineage_id: str
) -> Individual:
    from .models import Individual

    return Individual(
        [random_token(rng, macro_lib) for _ in range(rng.randint(10, 24))],
        rng.choice(CHARTS),
        lineage_id,
    )


def specialization_pair_bonus(a: str, b: str) -> float:
    if a == b:
        return 0.0
    good = {
        frozenset({"patch", "summary"}),
        frozenset({"patch", "plan"}),
        frozenset({"summary", "plan"}),
        frozenset({"generalist", "patch"}),
        frozenset({"generalist", "summary"}),
        frozenset({"generalist", "plan"}),
        frozenset({"summary_bridge", "summary"}),
        frozenset({"summary_bridge", "plan"}),
        frozenset({"summary_bridge", "patch"}),
        frozenset({"summary_bridge", "generalist"}),
    }
    return 1.0 if frozenset({a, b}) in good else 0.3


def build_protected_tokens(
    motif_archive: dict[str, dict[str, Any]], specialization: str
) -> set[str]:
    protected: set[str] = set()
    for motif, meta in motif_archive.items():
        if (
            meta["mean_stability"] >= 0.78
            and meta["specializations"].get(specialization, 0) >= 3
        ):
            protected.update(motif.split("|"))
    return protected


def _find_subsequence(dna: list[str], subsequence: Sequence[str]) -> int | None:
    if not subsequence or len(subsequence) > len(dna):
        return None
    for index in range(len(dna) - len(subsequence) + 1):
        if dna[index : index + len(subsequence)] == list(subsequence):
            return index
    return None


def _scrub_forbidden_subsequences(
    dna: list[str],
    rng: random.Random,
    macro_lib: Mapping[str, Sequence[str]],
    forbidden_subsequences: Iterable[Sequence[str]] | None,
    protected_tokens: set[str],
) -> list[str]:
    forbidden = [tuple(seq) for seq in forbidden_subsequences or () if len(seq) >= 2]
    if not forbidden:
        return dna

    cleaned = dna[:]
    for _ in range(max(2, len(cleaned))):
        changed = False
        for subsequence in forbidden:
            start = _find_subsequence(cleaned, subsequence)
            if start is None:
                continue
            changed = True
            mutable_positions = [
                start + offset
                for offset, token in enumerate(subsequence)
                if token not in protected_tokens
            ]
            if not mutable_positions:
                mutable_positions = list(range(start, start + len(subsequence)))
            cleaned[rng.choice(mutable_positions)] = random_token(rng, macro_lib)
        if not changed:
            break
    return cleaned


def scout_teleport(
    ind: Individual,
    rng: random.Random,
    macro_lib: Mapping[str, Sequence[str]],
    lineage_id: str,
    replace_fraction_min: float = 0.6,
    replace_fraction_max: float = 0.8,
) -> Individual:
    from .models import Individual

    dna = ind.dna[:]
    replace_fraction = rng.uniform(replace_fraction_min, replace_fraction_max)
    n_replace = max(1, int(len(dna) * replace_fraction))
    positions = rng.sample(range(len(dna)), min(n_replace, len(dna)))
    for position in positions:
        dna[position] = random_token(rng, macro_lib)

    if len(dna) < 8:
        dna.extend(random_token(rng, macro_lib) for _ in range(8 - len(dna)))

    child = Individual(dna[:36], rng.choice(CHARTS), lineage_id, [ind.lineage_id])
    child.parallax_role = "scout"
    child.scout_origin = ind.lineage_id
    child.valley_membership = "teleport"
    child.replication_gene = False
    child.replication_depth = 0
    child.replication_origin = ind.lineage_id
    return child


def replicate_scout(
    scout: Individual,
    rng: random.Random,
    macro_lib: Mapping[str, Sequence[str]],
    lineage_id: str,
    burst_size: int = 2,
) -> list[Individual]:
    from .models import Individual

    burst_size = max(1, min(3, burst_size))
    descendants: list[Individual] = []
    origin = getattr(scout, "scout_origin", scout.lineage_id)

    for index in range(burst_size):
        child_lineage = f"{lineage_id}:rep{index}"
        if index == 0:
            child = Individual(
                scout.dna[:], scout.chart, child_lineage, [scout.lineage_id]
            )
        else:
            intensity = 1.12 + 0.28 * index
            child = mutate(
                scout,
                rng,
                macro_lib,
                child_lineage,
                intensity=intensity,
            )
        child.parallax_role = "replica"
        child.scout_origin = origin
        child.valley_membership = "replication"
        child.replication_gene = True
        child.replication_depth = getattr(scout, "replication_depth", 0) + 1
        child.replication_origin = origin
        descendants.append(child)

    return descendants


def adversarial_push(
    ind: Individual,
    rng: random.Random,
    macro_lib: Mapping[str, Sequence[str]],
    lineage_id: str,
    gradient: dict[str, float] | None = None,
    push_strength: float = 0.5,
) -> Individual:
    from .models import Individual

    gradient = gradient or {}
    dna = ind.dna[:]
    positive_tokens = [token for token, delta in gradient.items() if delta > 0.08]
    negative_tokens = [token for token, delta in gradient.items() if delta < -0.08]

    if not positive_tokens or not negative_tokens:
        child = scout_teleport(ind, rng, macro_lib, lineage_id)
        child.parallax_role = "canary"
        child.valley_membership = "adversarial_fallback"
        child.gradient_signal = dict(gradient)
        return child

    target_positions = [
        index for index, token in enumerate(dna) if token in positive_tokens
    ]
    if not target_positions:
        target_positions = list(range(len(dna)))
    rng.shuffle(target_positions)

    mutation_budget = max(1, int(len(dna) * push_strength))
    for position in target_positions[:mutation_budget]:
        dna[position] = rng.choice(negative_tokens)

    if len(dna) > 8 and rng.random() < 0.35:
        del dna[rng.randrange(len(dna))]
    if len(dna) < 36 and rng.random() < 0.45:
        dna.insert(rng.randrange(len(dna) + 1), rng.choice(negative_tokens))

    child = Individual(dna[:36], ind.chart, lineage_id, [ind.lineage_id])
    child.parallax_role = "canary"
    child.valley_membership = "adversarial"
    child.gradient_signal = dict(gradient)
    return child


def mutate(
    ind: Individual,
    rng: random.Random,
    macro_lib: Mapping[str, Sequence[str]],
    lineage_id: str,
    intensity: float = 1.0,
    protected_tokens: set[str] | None = None,
    forbidden_subsequences: Iterable[Sequence[str]] | None = None,
    exploration_mode: str = "local",
    scout_replace_fraction_min: float = 0.6,
    scout_replace_fraction_max: float = 0.8,
) -> Individual:
    if exploration_mode == "scout":
        return scout_teleport(
            ind,
            rng,
            macro_lib,
            lineage_id,
            replace_fraction_min=scout_replace_fraction_min,
            replace_fraction_max=scout_replace_fraction_max,
        )

    dna = ind.dna[:]
    protected_tokens = protected_tokens or set()

    def mutable_positions():
        return [i for i, tok in enumerate(dna) if tok not in protected_tokens]

    n_ops = 1 + (1 if rng.random() < max(0.0, intensity - 1.0) else 0)
    for _ in range(n_ops):
        mpos = mutable_positions()
        action = rng.choice(["replace", "insert", "delete", "swap"])
        if action == "replace" and mpos:
            dna[rng.choice(mpos)] = random_token(rng, macro_lib)
        elif action == "insert" and len(dna) < 36:
            dna.insert(rng.randrange(len(dna) + 1), random_token(rng, macro_lib))
        elif action == "delete" and len(dna) > 8 and mpos:
            del dna[rng.choice(mpos)]
        elif action == "swap" and len(mpos) >= 2:
            a, b = rng.sample(mpos, 2)
            dna[a], dna[b] = dna[b], dna[a]

    dna = _scrub_forbidden_subsequences(
        dna,
        rng,
        macro_lib,
        forbidden_subsequences,
        protected_tokens,
    )

    chart = ind.chart if rng.random() < 0.8 else rng.choice(CHARTS)
    from .models import Individual

    child = Individual(dna, chart, lineage_id, [ind.lineage_id])
    child.parallax_role = getattr(ind, "parallax_role", "member")
    child.valley_membership = getattr(ind, "valley_membership", "")
    child.gradient_signal = dict(getattr(ind, "gradient_signal", {}))
    child.replication_gene = getattr(ind, "replication_gene", False)
    child.replication_depth = getattr(ind, "replication_depth", 0)
    child.replication_origin = getattr(
        ind, "replication_origin", getattr(ind, "scout_origin", "")
    )
    return child


def crossover(
    a: Individual,
    b: Individual,
    rng: random.Random,
    lineage_id: str,
) -> Individual:
    from .models import Individual

    if a.specialization == b.specialization:
        ia = rng.randrange(1, max(2, len(a.dna) - 1))
        ib = rng.randrange(ia, len(a.dna))
        donor_start = rng.randrange(0, max(1, len(b.dna) - (ib - ia)))
        donor_segment = b.dna[donor_start : donor_start + (ib - ia)]
        dna = a.dna[:ia] + donor_segment + a.dna[ib:]
    else:
        ia = rng.randrange(1, len(a.dna))
        ib = rng.randrange(1, len(b.dna))
        dna = (a.dna[:ia] + b.dna[ib:])[:36]
        if rng.random() < 0.35:
            donor_parent = (
                b
                if specialization_pair_bonus(a.specialization, b.specialization) >= 1.0
                else a
            )
            seg_len = min(4, len(donor_parent.dna))
            if seg_len >= 3 and len(dna) >= seg_len:
                s = rng.randrange(0, len(donor_parent.dna) - seg_len + 1)
                block = donor_parent.dna[s : s + seg_len]
                p = rng.randrange(0, max(1, len(dna) - seg_len + 1))
                dna[p : p + seg_len] = block

    dna = dna[:36]
    if len(dna) < 8:
        dna = (dna + a.dna)[:8]
    child = Individual(
        dna,
        a.chart if rng.random() < 0.5 else b.chart,
        lineage_id,
        [a.lineage_id, b.lineage_id],
    )
    child.replication_gene = getattr(a, "replication_gene", False) or getattr(
        b, "replication_gene", False
    )
    child.replication_depth = max(
        getattr(a, "replication_depth", 0), getattr(b, "replication_depth", 0)
    )
    child.replication_origin = getattr(
        a, "replication_origin", getattr(a, "scout_origin", "")
    ) or getattr(b, "replication_origin", getattr(b, "scout_origin", ""))
    return child
