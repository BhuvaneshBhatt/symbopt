"""Validated immutable controls for the global optimization pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .quadratic_models import QuadraticControls, quadratic_controls


class _Unset:
    def __repr__(self) -> str:
        return "<default>"


UNSET = _Unset()


@dataclass(frozen=True)
class SolverOptions:
    """Normalized solver controls shared by recursive optimization branches."""

    interval_max_boxes: int = 200
    interval_tol: float = 1e-3
    max_active_sets: int = 64
    use_sos: bool = True
    structural_rules: bool = True
    quadratic: QuadraticControls = field(default_factory=QuadraticControls)
    semialg_certification: str = "auto"
    semialg_range_cost_limit: int | None = None


def _mapping_section(options, name):
    values = options.get(name, {})
    if not isinstance(values, Mapping):
        raise TypeError(f"options[{name!r}] must be a mapping")
    return values


def normalize_solver_options(
    *,
    interval_max_boxes=UNSET,
    interval_tol=UNSET,
    max_active_sets=UNSET,
    use_sos=UNSET,
    semialg_certification=UNSET,
    semialg_range_cost_limit=UNSET,
    options: Mapping[str, Mapping[str, object]] | None = None,
) -> SolverOptions:
    """Merge flat and grouped controls, with explicitly supplied flat values winning."""
    grouped = {} if options is None else options
    if not isinstance(grouped, Mapping):
        raise TypeError("options must be a mapping of backend names to option mappings")
    allowed_sections = {"interval", "sos", "semialg", "kkt", "structural", "quadratic"}
    unknown_sections = set(grouped) - allowed_sections
    if unknown_sections:
        raise ValueError(
            "unknown option section(s): " + ", ".join(sorted(unknown_sections))
        )

    interval = _mapping_section(grouped, "interval")
    sos = _mapping_section(grouped, "sos")
    semialg = _mapping_section(grouped, "semialg")
    kkt = _mapping_section(grouped, "kkt")
    structural = _mapping_section(grouped, "structural")
    allowed = {
        "interval": {"max_boxes", "tol"},
        "sos": {"enabled"},
        "semialg": {"certification", "range_cost_limit"},
        "kkt": {"max_active_sets"},
        "structural": {"enabled"},
    }
    for name, section in (
        ("interval", interval),
        ("sos", sos),
        ("semialg", semialg),
        ("kkt", kkt),
        ("structural", structural),
    ):
        unknown = set(section) - allowed[name]
        if unknown:
            raise ValueError("unknown backend option(s): " + ", ".join(sorted(unknown)))

    def choose(explicit, section, key, default):
        if explicit is not UNSET:
            return explicit
        return section.get(key, default)

    max_boxes = choose(interval_max_boxes, interval, "max_boxes", 200)
    tol = choose(interval_tol, interval, "tol", 1e-3)
    active_sets = choose(max_active_sets, kkt, "max_active_sets", 64)
    sos_enabled = choose(use_sos, sos, "enabled", True)
    certification = choose(semialg_certification, semialg, "certification", "auto")
    range_limit = choose(semialg_range_cost_limit, semialg, "range_cost_limit", None)

    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes <= 0:
        raise ValueError("interval max_boxes must be a positive integer")
    try:
        tol = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("interval tol must be a positive finite number") from exc
    if not (tol > 0.0) or tol == float("inf"):
        raise ValueError("interval tol must be a positive finite number")
    if (
        isinstance(active_sets, bool)
        or not isinstance(active_sets, int)
        or active_sets < 0
    ):
        raise ValueError("max_active_sets must be a nonnegative integer")
    if not isinstance(sos_enabled, bool):
        raise TypeError("SOS enabled must be a boolean")
    if certification not in {"auto", "candidate", "complete"}:
        raise ValueError(
            "semialg certification must be 'auto', 'candidate', or 'complete'"
        )
    if range_limit is not None:
        if isinstance(range_limit, bool) or not isinstance(range_limit, int):
            raise TypeError(
                "semialg range_cost_limit must be a nonnegative integer or None"
            )
        if range_limit < 0:
            raise ValueError("semialg range_cost_limit must be nonnegative")

    rules_enabled = structural.get("enabled", True)
    if not isinstance(rules_enabled, bool):
        raise TypeError("structural enabled must be a boolean")
    return SolverOptions(
        structural_rules=rules_enabled,
        quadratic=quadratic_controls(**_mapping_section(grouped, "quadratic")),
        interval_max_boxes=max_boxes,
        interval_tol=tol,
        max_active_sets=active_sets,
        use_sos=sos_enabled,
        semialg_certification=certification,
        semialg_range_cost_limit=range_limit,
    )
