from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import sympy as sp
from sympy import Piecewise
from sympy.core.relational import (
    Equality,
    GreaterThan,
    LessThan,
    Relational,
    StrictGreaterThan,
    StrictLessThan,
)
from sympy.polys.polyerrors import CoercionFailed, PolynomialError
from sympy.polys.rootoftools import CRootOf, RootOf

from .algebraic_numbers import isolating_interval
from .constraint_reduction import reduce_lifted_problem
from .problem_models import LiftedProblem, LiftRecord


class _QuotientReducer:
    """Compile a quotient-ring Gröbner basis once and reuse it for reductions."""

    def __init__(self, equalities: Sequence[Relational], vars: Sequence[sp.Symbol]):
        self._basis = None
        polys = [
            sp.expand(rel.lhs - rel.rhs)
            for rel in equalities
            if isinstance(rel, Equality)
        ]
        if not polys or not vars:
            return
        try:
            self._basis = sp.groebner(polys, *vars, order="lex")
        except (
            PolynomialError,
            CoercionFailed,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            self._basis = None

    def reduce(self, expr: sp.Expr) -> sp.Expr:
        """Reduce an expression modulo the fixed finite-quotient relations."""
        expanded = sp.expand(expr)
        if self._basis is None:
            return expanded
        try:
            return sp.expand(self._basis.reduce(expanded)[1])
        except (
            PolynomialError,
            CoercionFailed,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            return expanded


def _canonicalize_expr(expr: sp.Expr) -> sp.Expr:
    try:
        expr = sp.together(expr)
    except (
        PolynomialError,
        CoercionFailed,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        pass
    try:
        expr = sp.cancel(expr)
    except (
        PolynomialError,
        CoercionFailed,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        pass
    try:
        expr = sp.powsimp(expr, force=True)
    except (
        PolynomialError,
        CoercionFailed,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        pass
    try:
        expr = sp.factor_terms(expr)
    except (
        PolynomialError,
        CoercionFailed,
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
    ):
        pass
    return sp.expand(expr)


def _dedupe_relations(rels):
    out = []
    seen = set()
    for rel in rels:
        key = sp.srepr(rel)
        if key not in seen:
            seen.add(key)
            out.append(rel)
    return out


class Algebraizer:
    """Convert supported SymPy expressions into polynomial lifted problems.

    The algebraizer introduces auxiliary variables together with graph, domain,
    and branch constraints. Instances retain caches so repeated subexpressions share
    lifts within one transformation."""

    TRANSCENDENTAL_FUNCS: ClassVar[set[object]] = {
        sp.exp,
        sp.log,
        sp.sin,
        sp.cos,
        sp.tan,
        sp.cot,
        sp.sec,
        sp.csc,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.asin,
        sp.acos,
        sp.atan,
        sp.sqrt,
    }

    def __init__(
        self,
        aux_prefix: str = "aux",
        allow_transcendentals: bool = True,
        allow_rootof: bool = True,
        assumption_aware: bool = True,
        reduce_after_lift: bool = True,
    ) -> None:
        self.aux_prefix = aux_prefix
        self.allow_transcendentals = allow_transcendentals
        self.allow_rootof = allow_rootof
        self.assumption_aware = assumption_aware
        self.reduce_after_lift = reduce_after_lift

        self._counter = 0
        self._reserved_names = set()
        self._memo: dict[sp.Expr, sp.Expr] = {}

        self.defining_equalities: list[Equality] = []
        self.branch_inequalities: list[GreaterThan] = []
        self.strict_inequalities: list[StrictGreaterThan] = []
        self.graph_equalities: list[Equality] = []
        self.graph_inequalities: list[Relational] = []
        self.auxiliary_variables: list[sp.Symbol] = []
        self.substitution_map: dict[sp.Expr, sp.Symbol] = {}
        self.lift_records: list[LiftRecord] = []
        self.preprocessing_notes: list[str] = []
        self._trig_cache: dict[sp.Expr, dict[str, sp.Symbol]] = {}
        self._hyper_cache: dict[sp.Expr, dict[str, sp.Symbol]] = {}
        self._inverse_cache: dict[sp.Expr, sp.Symbol] = {}
        self._root_cache: dict[tuple[sp.Expr, int], sp.Symbol] = {}
        self._rootof_cache: dict[sp.Expr, sp.Symbol] = {}

    def _new_aux(self) -> sp.Symbol:
        self._counter += 1
        while f"{self.aux_prefix}_{self._counter}" in self._reserved_names:
            self._counter += 1
        name = f"{self.aux_prefix}_{self._counter}"
        self._reserved_names.add(name)
        v = sp.Symbol(name, real=True)
        self.auxiliary_variables.append(v)
        return v

    def _register_substitution(
        self, original_expr: sp.Expr, aux_var: sp.Symbol
    ) -> None:
        self.substitution_map.setdefault(original_expr, aux_var)

    def _record_lift(
        self,
        aux: sp.Symbol,
        original_expr: sp.Expr,
        kind: str,
        defining_eqs: Sequence[Relational] = (),
        domain_rels: Sequence[Relational] = (),
        dependencies: Sequence[sp.Symbol] = (),
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.lift_records.append(
            LiftRecord(
                aux=aux,
                original_expr=original_expr,
                kind=kind,
                defining_eqs=list(defining_eqs),
                domain_rels=list(domain_rels),
                dependencies=list(dependencies),
                metadata=dict(metadata or {}),
            )
        )

    def _add_eq_zero(self, expr: sp.Expr) -> Equality:
        rel = sp.Eq(sp.expand(expr), 0)
        self.defining_equalities.append(rel)
        return rel

    def _add_ge_zero(self, expr: sp.Expr) -> GreaterThan:
        rel = sp.Ge(sp.expand(expr), 0)
        self.branch_inequalities.append(rel)
        return rel

    def _add_gt_zero(self, expr: sp.Expr) -> StrictGreaterThan:
        rel = sp.Gt(sp.expand(expr), 0)
        self.strict_inequalities.append(rel)
        return rel

    def _add_graph_eq(self, lhs: sp.Expr, rhs: sp.Expr) -> Equality:
        rel = sp.Eq(lhs, rhs)
        self.graph_equalities.append(rel)
        return rel

    def _add_graph_rel(self, rel: Relational) -> None:
        self.graph_inequalities.append(rel)

    def _preprocess_expr(self, expr: sp.Expr) -> sp.Expr:
        original = expr
        expr = _canonicalize_expr(expr)
        if expr != original:
            self.preprocessing_notes.append(
                f"preprocessed: {sp.sstr(original)} -> {sp.sstr(expr)}"
            )
        return expr

    def _preprocess_constraints(
        self, constraints: Sequence[Relational]
    ) -> list[Relational]:
        out: list[Relational] = []
        for rel in constraints:
            if rel is sp.S.true or rel is True:
                continue
            if rel is sp.S.false or rel is False:
                out.append(sp.Eq(sp.Integer(1), sp.Integer(0), evaluate=False))
                continue
            lhs = self._preprocess_expr(rel.lhs)
            rhs = self._preprocess_expr(rel.rhs)
            out.append(rel.func(lhs, rhs))
        return out

    def algebraize_problem(
        self,
        objective: sp.Expr,
        constraints: Sequence[Relational] | None = None,
        variables: Sequence[sp.Symbol] | None = None,
    ) -> LiftedProblem:
        """Lift a symbolic optimization problem into exact algebraic constraints."""
        raw_constraints = list(constraints or [])
        raw_symbols = set(objective.free_symbols)
        for relation in raw_constraints:
            if isinstance(relation, Relational):
                raw_symbols.update(relation.free_symbols)
        if variables is None:
            original_vars = sorted(raw_symbols, key=lambda item: item.sort_key())
        else:
            original_vars = list(variables)
        parameter_symbols = sorted(
            raw_symbols.difference(original_vars), key=lambda item: item.sort_key()
        )
        self._reserved_names.update(
            symbol.name for symbol in (*raw_symbols, *original_vars)
        )

        # Polynomial relational inputs can be represented directly; other
        # inputs continue through canonicalization and lifting below.
        if (
            all(isinstance(relation, Relational) for relation in raw_constraints)
            and not any(
                isinstance(node, (RootOf, CRootOf))
                for node in sp.preorder_traversal(objective)
            )
            and objective.is_polynomial(*original_vars)
            and all(
                sp.expand(relation.lhs - relation.rhs).is_polynomial(*original_vars)
                for relation in raw_constraints
            )
        ):
            lifted_constraints = []
            for relation in raw_constraints:
                residual = sp.expand(relation.lhs - relation.rhs)
                if isinstance(relation, Equality):
                    lifted_constraints.append(sp.Eq(residual, 0))
                elif isinstance(relation, GreaterThan):
                    lifted_constraints.append(sp.Ge(residual, 0))
                elif isinstance(relation, StrictGreaterThan):
                    lifted_constraints.append(sp.Gt(residual, 0))
                elif isinstance(relation, LessThan):
                    lifted_constraints.append(sp.Ge(-residual, 0))
                elif isinstance(relation, StrictLessThan):
                    lifted_constraints.append(sp.Gt(-residual, 0))
                elif isinstance(relation, sp.Unequality):
                    lifted_constraints.append(sp.Ne(residual, 0))
                else:
                    lifted_constraints.append(relation)
            strict_warnings = []
            if any(
                isinstance(rel, (StrictGreaterThan, StrictLessThan))
                for rel in raw_constraints
            ):
                strict_warnings.append(
                    "Strict inequalities are preserved symbolically. Downstream optimizations that require closed feasible sets may use the closure_constraints() view, which can certify only the closure and may distinguish minima from infima imperfectly."
                )
            fast_problem = LiftedProblem(
                original_objective=objective,
                lifted_objective=sp.expand(objective),
                original_constraints=list(raw_constraints),
                lifted_constraints=lifted_constraints,
                defining_equalities=[],
                branch_inequalities=[],
                strict_inequalities=[],
                graph_equalities=[],
                graph_inequalities=[],
                auxiliary_variables=[],
                substitution_map={},
                all_variables=list(original_vars),
                lift_records=[],
                preprocessing_notes=["polynomial fast path"],
                strict_closure_warnings=strict_warnings,
                original_variables=list(original_vars),
                parameter_symbols=list(parameter_symbols),
                polynomial_problem=True,
                structural_metadata={"algebraization_fast_path": "polynomial"},
            )
            fast_problem, _ = reduce_lifted_problem(fast_problem)
            return fast_problem

        constraints = self._preprocess_constraints(raw_constraints)
        objective = self._preprocess_expr(objective)

        lifted_objective = self.algebraize_expr(objective)
        lifted_constraints = [self.algebraize_relational(rel) for rel in constraints]

        processed_symbols = objective.free_symbols.union(
            *(rel.free_symbols for rel in constraints)
        )
        if variables is None:
            original_vars = sorted(processed_symbols, key=lambda item: item.sort_key())
        else:
            original_vars = list(variables)
        parameter_symbols = sorted(
            processed_symbols.difference(original_vars),
            key=lambda item: item.sort_key(),
        )
        all_vars = list(original_vars) + list(self.auxiliary_variables)

        strict_closure_warnings: list[str] = []
        if self.reduce_after_lift and all_vars:
            quotient_reducer = _QuotientReducer(self.defining_equalities, all_vars)
            reduce_expr = quotient_reducer.reduce
            lifted_objective = reduce_expr(lifted_objective)

            reduced_constraints = []
            for rel in lifted_constraints:
                lhs = reduce_expr(rel.lhs)
                rhs = reduce_expr(rel.rhs)
                reduced_constraints.append(rel.func(lhs, rhs))
            lifted_constraints = reduced_constraints

            self.graph_equalities = [
                sp.Eq(reduce_expr(rel.lhs), reduce_expr(rel.rhs))
                for rel in self.graph_equalities
            ]

            reduced_graph_ineqs = []
            for rel in self.graph_inequalities:
                if hasattr(rel, "lhs") and hasattr(rel, "rhs"):
                    reduced_graph_ineqs.append(
                        rel.func(
                            reduce_expr(rel.lhs),
                            reduce_expr(rel.rhs),
                        )
                    )
                else:
                    reduced_graph_ineqs.append(rel)
            self.graph_inequalities = reduced_graph_ineqs

            self.branch_inequalities = [
                sp.Ge(reduce_expr(rel.lhs), reduce_expr(rel.rhs))
                for rel in self.branch_inequalities
            ]
            self.strict_inequalities = [
                sp.Gt(reduce_expr(rel.lhs), reduce_expr(rel.rhs))
                for rel in self.strict_inequalities
            ]

        if self.strict_inequalities:
            strict_closure_warnings.append(
                "Strict inequalities are preserved symbolically. Downstream optimizations that require closed feasible sets may use the closure_constraints() view, which can certify only the closure and may distinguish minima from infima imperfectly."
            )

        self.defining_equalities = _dedupe_relations(self.defining_equalities)
        self.branch_inequalities = _dedupe_relations(self.branch_inequalities)
        self.strict_inequalities = _dedupe_relations(self.strict_inequalities)
        self.graph_equalities = _dedupe_relations(self.graph_equalities)
        self.graph_inequalities = _dedupe_relations(self.graph_inequalities)
        problem_out = LiftedProblem(
            original_objective=objective,
            lifted_objective=sp.expand(lifted_objective),
            original_constraints=list(constraints),
            lifted_constraints=lifted_constraints,
            defining_equalities=list(self.defining_equalities),
            branch_inequalities=list(self.branch_inequalities),
            strict_inequalities=list(self.strict_inequalities),
            graph_equalities=list(self.graph_equalities),
            graph_inequalities=list(self.graph_inequalities),
            auxiliary_variables=list(self.auxiliary_variables),
            substitution_map=dict(self.substitution_map),
            all_variables=all_vars,
            lift_records=list(self.lift_records),
            preprocessing_notes=list(self.preprocessing_notes),
            strict_closure_warnings=strict_closure_warnings,
            original_variables=list(original_vars),
            parameter_symbols=list(parameter_symbols),
        )
        problem_out, _ = reduce_lifted_problem(problem_out)
        return problem_out

    def algebraize_expr(self, expr: sp.Expr) -> sp.Expr:
        """Lift a supported expression and retain its defining graph and domain restrictions."""
        expr = self._preprocess_expr(expr)
        if expr in self._memo:
            return self._memo[expr]

        if isinstance(expr, sp.AlgebraicNumber):
            result = self.algebraize_expr(expr.as_expr())
            self._memo[expr] = result
            return result

        if expr.is_Atom:
            self._memo[expr] = expr
            return expr

        if isinstance(expr, sp.Piecewise):
            raise NotImplementedError(
                "General Piecewise expressions require case splitting. Use algebraize_cases(...) to branch them."
            )

        if expr.func is sp.Abs:
            result = self._algebraize_abs(expr)
            self._memo[expr] = result
            return result

        if expr.func is sp.Pow:
            result = self._algebraize_pow(expr)
            self._memo[expr] = result
            return result

        if isinstance(expr, (RootOf, CRootOf)):
            result = self._algebraize_rootof(expr)
            self._memo[expr] = result
            return result

        if expr.is_Add or expr.is_Mul:
            new_args = [self.algebraize_expr(a) for a in expr.args]
            result = _canonicalize_expr(expr.func(*new_args))
            self._memo[expr] = result
            return result

        if expr.is_Function:
            if expr.func in self.TRANSCENDENTAL_FUNCS:
                if not self.allow_transcendentals:
                    raise NotImplementedError(
                        f"Transcendental lifting disabled for {expr.func.__name__}"
                    )
                result = self._lift_transcendental(expr)
                self._memo[expr] = result
                return result
            raise NotImplementedError(f"Unsupported function: {expr.func.__name__}")

        new_args = [self.algebraize_expr(a) for a in expr.args]
        result = expr.func(*new_args)
        self._memo[expr] = result
        return result

    def _normalize_rel_expr(self, expr: sp.Expr) -> sp.Expr:
        expr = _canonicalize_expr(expr)
        try:
            num, den = sp.fraction(sp.together(expr))
            if den.is_positive is True:
                expr = sp.expand(num)
            elif den.is_negative is True:
                expr = sp.expand(-num)
            else:
                expr = sp.expand(num / den)
        except (
            PolynomialError,
            CoercionFailed,
            TypeError,
            ValueError,
            ArithmeticError,
            NotImplementedError,
        ):
            expr = sp.expand(expr)
        return expr

    def algebraize_relational(self, rel: Relational) -> Relational:
        """Lift a relation without discarding the original inequality orientation or source guards."""
        lhs = self.algebraize_expr(rel.lhs)
        rhs = self.algebraize_expr(rel.rhs)
        if isinstance(rel, Equality):
            return sp.Eq(self._normalize_rel_expr(lhs - rhs), 0)
        if isinstance(rel, GreaterThan):
            return sp.Ge(self._normalize_rel_expr(lhs - rhs), 0)
        if isinstance(rel, StrictGreaterThan):
            return sp.Gt(self._normalize_rel_expr(lhs - rhs), 0)
        if isinstance(rel, LessThan):
            return sp.Ge(self._normalize_rel_expr(rhs - lhs), 0)
        if isinstance(rel, StrictLessThan):
            return sp.Gt(self._normalize_rel_expr(rhs - lhs), 0)
        if isinstance(rel, sp.Unequality):
            return sp.Ne(self._normalize_rel_expr(lhs - rhs), 0)
        raise TypeError(f"Unsupported relational type: {type(rel)}")

    def _algebraize_abs(self, expr: sp.Abs) -> sp.Symbol:
        inner = self.algebraize_expr(expr.args[0])
        aux = self._new_aux()
        self._register_substitution(expr, aux)
        eq = self._add_eq_zero(aux**2 - inner**2)
        ge = self._add_ge_zero(aux)
        self._record_lift(
            aux,
            expr,
            "abs",
            [eq],
            [ge],
            [d for d in inner.free_symbols if d in self.auxiliary_variables],
        )
        return aux

    def _root_domain_rels(
        self, base_poly: sp.Expr, denominator: int
    ) -> list[Relational]:
        rels: list[Relational] = []
        if denominator % 2 == 0:
            if self.assumption_aware:
                if base_poly.is_nonnegative is not True:
                    rels.append(sp.Ge(sp.expand(base_poly), 0))
            else:
                rels.append(sp.Ge(sp.expand(base_poly), 0))
        return rels

    def _algebraize_pow(self, expr: sp.Pow) -> sp.Expr:
        base = self.algebraize_expr(expr.base)
        exponent = expr.exp

        if exponent.is_Integer:
            n = int(exponent)
            if n >= 0:
                return sp.expand(base**n)
            return self._introduce_inverse(
                sp.expand(base ** (-n)), expr, denominator_expr=base
            )

        if exponent.is_Rational:
            rat = sp.Rational(exponent)
            q = int(rat.q)
            p = int(rat.p)
            if q == 1:
                if p >= 0:
                    return sp.expand(base**p)
                return self._introduce_inverse(
                    sp.expand(base ** (-p)), expr, denominator_expr=base
                )

            root_var = self._introduce_root(base, q, expr)
            poly = sp.expand(root_var ** abs(p))
            if p >= 0:
                return poly
            return self._introduce_inverse(poly, expr, denominator_expr=root_var)

        raise NotImplementedError(f"Unsupported exponent type: {sp.sstr(expr)}")

    def _introduce_root(
        self, base_poly: sp.Expr, denominator: int, original_expr: sp.Expr
    ) -> sp.Symbol:
        key = (sp.expand(base_poly), int(denominator))
        if key in self._root_cache:
            root_var = self._root_cache[key]
            self._register_substitution(original_expr, root_var)
            return root_var
        root_var = self._new_aux()
        self._root_cache[key] = root_var
        self._register_substitution(original_expr, root_var)
        eq = self._add_eq_zero(root_var**denominator - base_poly)
        domain_rels: list[Relational] = []
        if denominator % 2 == 0:
            domain_rels.append(self._add_ge_zero(root_var))
            for rel in self._root_domain_rels(base_poly, denominator):
                if isinstance(rel, GreaterThan):
                    self.branch_inequalities.append(rel)
                else:
                    self.graph_inequalities.append(rel)
                domain_rels.append(rel)
        self._record_lift(
            root_var,
            original_expr,
            "root",
            [eq],
            domain_rels,
            [d for d in base_poly.free_symbols if d in self.auxiliary_variables],
            {"denominator": denominator, "principal_real_branch": denominator % 2 == 0},
        )
        return root_var

    def _introduce_inverse(
        self,
        poly_expr: sp.Expr,
        original_expr: sp.Expr,
        denominator_expr: sp.Expr | None = None,
    ) -> sp.Symbol:
        key = sp.expand(poly_expr)
        if key in self._inverse_cache:
            inv_var = self._inverse_cache[key]
            self._register_substitution(original_expr, inv_var)
            return inv_var
        inv_var = self._new_aux()
        self._inverse_cache[key] = inv_var
        self._register_substitution(original_expr, inv_var)
        eq = self._add_eq_zero(inv_var * poly_expr - 1)
        nonzero_rel = sp.Ne(sp.expand(poly_expr), 0)
        self._add_graph_rel(nonzero_rel)
        if denominator_expr is not None:
            self._add_graph_rel(sp.Ne(sp.expand(denominator_expr), 0))
        self._record_lift(
            inv_var,
            original_expr,
            "inverse",
            [eq],
            [nonzero_rel],
            [d for d in poly_expr.free_symbols if d in self.auxiliary_variables],
            {
                "denominator_expr": sp.sstr(denominator_expr)
                if denominator_expr is not None
                else sp.sstr(poly_expr)
            },
        )
        return inv_var

    def _algebraize_rootof(self, expr: sp.Expr) -> sp.Symbol:
        if not self.allow_rootof:
            raise NotImplementedError("RootOf / CRootOf lifting disabled.")
        if expr in self._rootof_cache:
            aux = self._rootof_cache[expr]
            self._register_substitution(expr, aux)
            return aux

        poly_expr, _ = expr.args
        poly_expr = sp.sympify(poly_expr)
        symbols = list(poly_expr.free_symbols)
        if len(symbols) != 1:
            raise NotImplementedError(
                "Only univariate RootOf / CRootOf objects are supported."
            )
        z = symbols[0]

        aux = self._new_aux()
        self._rootof_cache[expr] = aux
        self._register_substitution(expr, aux)
        eq = self._add_eq_zero(poly_expr.subs(z, aux))
        domain_rels: list[Relational] = []

        if getattr(expr, "is_real", False):
            interval = isolating_interval(expr)
            if interval is None:
                raise NotImplementedError(
                    "Could not obtain an exact isolating interval for RootOf branch."
                )
            a, b = interval
            domain_rels.append(self._add_ge_zero(aux - a))
            domain_rels.append(self._add_ge_zero(b - aux))
            if a == b:
                domain_rels.append(self._add_eq_zero(aux - a))
        else:
            raise NotImplementedError("Only real RootOf branches are supported.")

        self._record_lift(
            aux,
            expr,
            "rootof",
            [eq],
            domain_rels,
            [],
            {
                "parametric": bool(expr.free_symbols),
                "isolated_interval": len(domain_rels) >= 2,
                "isolating_interval": tuple(map(sp.sstr, interval)),
                "exact_branch": True,
            },
        )
        return aux

    def _add_identity_rels(self, func: object, arg: sp.Expr, aux: sp.Symbol) -> None:
        if func in (sp.sin, sp.cos):
            cache = self._trig_cache.setdefault(arg, {})
            key = "sin" if func is sp.sin else "cos"
            cache[key] = aux
            if "sin" in cache and "cos" in cache:
                self._add_eq_zero(cache["sin"] ** 2 + cache["cos"] ** 2 - 1)
        if func in (sp.sinh, sp.cosh):
            cache = self._hyper_cache.setdefault(arg, {})
            key = "sinh" if func is sp.sinh else "cosh"
            cache[key] = aux
            if "sinh" in cache and "cosh" in cache:
                self._add_eq_zero(cache["cosh"] ** 2 - cache["sinh"] ** 2 - 1)

    def _lift_transcendental(self, expr: sp.Expr) -> sp.Symbol:
        lifted_args = [self.algebraize_expr(arg) for arg in expr.args]
        aux = self._new_aux()
        self._register_substitution(expr, aux)
        rebuilt = expr.func(*lifted_args)
        eq = self._add_graph_eq(aux, rebuilt)
        domain_rels: list[Relational] = []

        if expr.func is sp.log:
            rel = sp.Gt(lifted_args[0], 0)
            self._add_graph_rel(rel)
            domain_rels.append(rel)
        if expr.func is sp.exp:
            rel = sp.Gt(aux, 0)
            self._add_graph_rel(rel)
            domain_rels.append(rel)
        if expr.func is sp.cosh:
            rel = sp.Ge(aux, 1)
            self._add_graph_rel(rel)
            domain_rels.append(rel)
        if expr.func is sp.tanh:
            rel1 = sp.Gt(aux + 1, 0)
            rel2 = sp.Gt(1 - aux, 0)
            self._add_graph_rel(rel1)
            self._add_graph_rel(rel2)
            domain_rels.extend([rel1, rel2])
        if expr.func in (sp.tan, sp.cot, sp.sec, sp.csc):
            arg = lifted_args[0]
            c = self._new_aux()
            s = self._new_aux()
            self._add_graph_eq(c, sp.cos(arg))
            self._add_graph_eq(s, sp.sin(arg))
            self._add_eq_zero(c**2 + s**2 - 1)
            if expr.func is sp.tan:
                self._add_graph_eq(aux * c, s)
                denominator = c
            elif expr.func is sp.cot:
                self._add_graph_eq(aux * s, c)
                denominator = s
            elif expr.func is sp.sec:
                self._add_graph_eq(aux * c, 1)
                denominator = c
            else:
                self._add_graph_eq(aux * s, 1)
                denominator = s
            ne = sp.Ne(denominator, 0)
            self._add_graph_rel(ne)
            domain_rels.append(ne)
        if expr.func is sp.asin:
            rel1 = sp.Ge(lifted_args[0] + 1, 0)
            rel2 = sp.Ge(1 - lifted_args[0], 0)
            self._add_graph_rel(rel1)
            self._add_graph_rel(rel2)
            domain_rels.extend([rel1, rel2])
        if expr.func is sp.acos:
            rel1 = sp.Ge(lifted_args[0] + 1, 0)
            rel2 = sp.Ge(1 - lifted_args[0], 0)
            self._add_graph_rel(rel1)
            self._add_graph_rel(rel2)
            domain_rels.extend([rel1, rel2])

        if lifted_args:
            self._add_identity_rels(expr.func, lifted_args[0], aux)
        self._record_lift(
            aux,
            expr,
            "transcendental_graph",
            [eq],
            domain_rels,
            [d for d in rebuilt.free_symbols if d in self.auxiliary_variables],
            {"function": expr.func.__name__},
        )
        return aux


def _branch_relations(condition):
    """Return conjunctive relational branch conditions, or None if unsupported."""
    condition = sp.sympify(condition)
    if condition is sp.S.true:
        return []
    if isinstance(condition, Relational):
        return [condition]
    if isinstance(condition, sp.And) and all(
        isinstance(term, Relational) for term in condition.args
    ):
        return list(condition.args)
    return None


def _piecewise_expr_cases(expr: sp.Expr) -> list[tuple[list[Relational], sp.Expr]]:
    """Expand Piecewise expressions into mutually exclusive relational cases."""
    if not expr.has(Piecewise):
        return [([], expr)]
    expr = sp.piecewise_exclusive(expr, deep=True)
    atoms = sorted(expr.atoms(Piecewise), key=sp.sstr)
    branches: list[tuple[list[Relational], sp.Expr]] = [([], expr)]
    for pw in atoms:
        new_branches: list[tuple[list[Relational], sp.Expr]] = []
        for assumptions, current_expr in branches:
            for piece_expr, condition in pw.args:
                relations = _branch_relations(condition)
                if relations is None:
                    continue
                new_branches.append(
                    (
                        [*assumptions, *relations],
                        sp.simplify(current_expr.xreplace({pw: piece_expr})),
                    )
                )
        branches = new_branches
    return branches


def algebraize_cases(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    aux_prefix: str = "aux",
    allow_transcendentals: bool = True,
    allow_rootof: bool = True,
    variables: Sequence[sp.Symbol] | None = None,
) -> list[LiftedProblem]:
    """Return lifted branch cases for objectives or constraints containing branching structure.

    Each returned problem carries the branch conditions needed to preserve the
    semantics of ``Piecewise`` and related branch-producing expressions."""
    constraints = list(constraints or [])
    objective_cases = _piecewise_expr_cases(objective)
    all_cases: list[LiftedProblem] = []
    case_no = 0
    for objective_assumptions, objective_expr in objective_cases:
        current = [(list(objective_assumptions), [], objective_expr)]
        for rel in constraints:
            lhs_cases = _piecewise_expr_cases(rel.lhs)
            rhs_cases = _piecewise_expr_cases(rel.rhs)
            nxt = []
            for ass0, cons0, obj0 in current:
                for la, le in lhs_cases:
                    for ra, re in rhs_cases:
                        nxt.append((ass0 + la + ra, cons0 + [rel.func(le, re)], obj0))
            current = nxt
        for assumptions, cons, obj in current:
            case_no += 1
            engine = Algebraizer(
                aux_prefix=f"{aux_prefix}_branch{case_no}",
                allow_transcendentals=allow_transcendentals,
                allow_rootof=allow_rootof,
            )
            lifted = engine.algebraize_problem(
                obj, list(cons) + list(assumptions), variables=variables
            )
            all_cases.append(lifted)
    return all_cases


def algebraize(
    objective: sp.Expr,
    constraints: Sequence[Relational] | None = None,
    aux_prefix: str = "aux",
    allow_transcendentals: bool = True,
    allow_rootof: bool = True,
    variables: Sequence[sp.Symbol] | None = None,
) -> LiftedProblem:
    """Lift one objective and constraint set into a ``LiftedProblem``.

    Parameters
    ----------
    objective:
        SymPy objective expression.
    constraints:
        Relational constraints in the original variables.
    allow_transcendentals, allow_rootof:
        Enable the corresponding supported lift families.

    Returns
    -------
    LiftedProblem
        Polynomial/algebraic representation plus reconstruction metadata."""
    engine = Algebraizer(
        aux_prefix=aux_prefix,
        allow_transcendentals=allow_transcendentals,
        allow_rootof=allow_rootof,
    )
    return engine.algebraize_problem(objective, constraints, variables=variables)
