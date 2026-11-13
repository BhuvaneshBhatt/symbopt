from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticKind(str, Enum):
    """Categories used to distinguish unsupported, symbolic, numerical, certificate, and internal failures."""

    UNSUPPORTED = "unsupported"
    SYMBOLIC_FAILURE = "symbolic_failure"
    NUMERICAL_FAILURE = "numerical_failure"
    CERTIFICATE_FAILURE = "certificate_failure"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class SolverDiagnostic:
    """Structured record of a solver-stage failure or fallback.

    Diagnostics are attached to ``PipelineResult`` so users can distinguish a
    normal unsupported-method fallback from an unexpected internal failure."""

    solver: str
    stage: str
    kind: DiagnosticKind
    message: str
    exception_type: str | None = None

    def pretty(self) -> str:
        """Format the stored mathematical result and diagnostic metadata for inspection."""
        detail = f" ({self.exception_type})" if self.exception_type else ""
        return f"{self.kind.value}: {self.solver}/{self.stage}: {self.message}{detail}"

    @classmethod
    def from_exception(
        cls,
        solver: str,
        stage: str,
        kind: DiagnosticKind,
        exc: BaseException,
    ) -> SolverDiagnostic:
        """Record a backend exception with its solver, stage and diagnostic category."""
        return cls(
            solver=solver,
            stage=stage,
            kind=kind,
            message=str(exc) or exc.__class__.__name__,
            exception_type=exc.__class__.__name__,
        )
