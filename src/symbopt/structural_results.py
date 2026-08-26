from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import sympy as sp


@dataclass(frozen=True)
class StructuralGlobalResult:
    """Globally justified result returned by a specialized solver route."""

    kind: str
    value: sp.Expr
    points: Tuple[Mapping[sp.Symbol, sp.Expr], ...] = ()
    attained: bool = False
    certified: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)
