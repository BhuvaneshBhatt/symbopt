from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import sympy as sp


@dataclass(frozen=True)
class StructuralGlobalResult:
    """Globally justified result returned by a specialized solver route."""

    kind: str
    value: sp.Expr
    points: tuple[Mapping[sp.Symbol, sp.Expr], ...] = ()
    attained: bool | None = False
    certified: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)
