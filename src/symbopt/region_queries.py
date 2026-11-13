"""Exact optimizer sets and parameter-dependent extremum relations."""

from dataclasses import dataclass

import sympy as sp

from .input_normalization import (
    normalize_bounds_input,
    normalize_constraint_branches,
    normalize_variables,
)
from .rational_optimization import preserve_rational_domain
from .region_inputs import prepare_region
from .region_optimization import constraint_formula


def _problem(objective, constraints, variables, region, bounds):
    prepared = None if region is None else prepare_region(region, variables)
    coordinates = (
        list(prepared.variables)
        if prepared is not None
        else normalize_variables(variables)
    )
    if coordinates is None:
        raise ValueError("exact region queries require explicit decision variables")
    formula = constraint_formula(constraints)
    if (
        sp.sympify(objective).has(sp.Float)
        or formula.has(sp.Float)
        or (prepared is not None and prepared.source_formula.has(sp.Float))
    ):
        from .region_inputs import RegionConversionError

        raise RegionConversionError(
            "admission", "optimizer formulas require exact coefficients"
        )
    if prepared is not None:
        formula = sp.And(formula, prepared.membership)
    additional, _ = normalize_bounds_input(bounds, coordinates)
    if formula is sp.false:
        formula = sp.Eq(0, 1, evaluate=False)
    branches = normalize_constraint_branches([formula, *additional])
    domain = sp.Or(
        *(
            sp.And(
                *preserve_rational_domain(sp.sympify(objective), branch, coordinates)
            )
            for branch in branches
        )
    )
    return sp.sympify(objective), domain, tuple(coordinates), prepared


def _eliminate(formula, coordinates, prepared):
    from semialg import SemialgebraicRegion

    from .region_inputs import RegionConversionError

    if formula.has(sp.Float):
        raise RegionConversionError(
            "elimination", "exact QE requires exact coefficients"
        )
    if prepared is not None:
        prepared.check_budget(formula)
    return sp.simplify(
        SemialgebraicRegion(formula, coordinates).quantifier_free_formula()
    )


def optimizer_locus(
    objective,
    constraints=None,
    *,
    variables=None,
    region=None,
    bounds=None,
    sense="minimize",
    eliminate=True,
):
    """Return the exact formula for all global optimizers, including continua.

    A completed value proof gives the feasible level set immediately. Otherwise
    universal comparison defines the locus, and optional real QE removes its
    binders. ``eliminate=False`` returns that exact logical definition without
    claiming a completed elimination or a numerical solution.
    """
    if sense not in {"minimize", "maximize"}:
        raise ValueError("sense must be minimize or maximize")
    objective, domain, coordinates, prepared = _problem(
        objective, constraints, variables, region, bounds
    )
    from .solver_pipeline import maximize_result, minimize_result

    if eliminate:
        solve = minimize_result if sense == "minimize" else maximize_result
        result = solve(
            objective,
            sp.Eq(0, 1, evaluate=False) if domain is sp.false else domain,
            variables=coordinates,
        )
        if result.optimizer_locus is not None:
            return result.optimizer_locus
    from semialg.quantifiers import ForAll

    other = tuple(sp.Dummy("competitor", real=True) for _ in coordinates)
    rename = dict(zip(coordinates, other))
    comparison = (
        objective <= objective.xreplace(rename)
        if sense == "minimize"
        else objective >= objective.xreplace(rename)
    )
    formula = sp.And(
        domain, ForAll(other, sp.Or(sp.Not(domain.xreplace(rename)), comparison))
    )
    return _eliminate(formula, coordinates, prepared) if eliminate else formula


def _eliminate_family(formula, coordinates, parameters, prepared):
    from .input_normalization import assumption_constraints

    rename = {p: sp.Dummy(p.name, real=True) for p in parameters}
    guards = assumption_constraints(parameters)
    guards += [sp.Eq(p, 0, evaluate=False) for p in parameters if p.is_zero is True]
    guards += [sp.Ne(p, 0, evaluate=False) for p in parameters if p.is_nonzero is True]
    condition = sp.And(*(guard.xreplace(rename) for guard in guards))
    converted = sp.And(condition, formula.xreplace(rename))
    result = _eliminate(
        converted, tuple(rename.get(v, v) for v in coordinates), prepared
    )
    return result.xreplace({v: p for p, v in rename.items()})


@dataclass(frozen=True)
class OptimizationFamily:
    """Exact extremum, attainment and optimizer relations over coefficient parameters.

    Value relations describe finite infima/suprema. Empty and unbounded fibers
    have separate conditions. Quantified definitions remain exact logical
    formulas when QE is not requested; they are not numerical answer tables.
    """

    objective: sp.Expr
    variables: tuple
    parameters: tuple
    value_symbol: sp.Symbol
    value_relation: sp.Expr
    optimizer_formula: sp.Expr
    attained_condition: sp.Expr
    feasible_condition: sp.Expr
    unbounded_condition: sp.Expr
    topology: object
    quantifier_free: bool


def optimization_family(
    objective,
    constraints=None,
    *,
    variables=None,
    parameters,
    region=None,
    bounds=None,
    sense="minimize",
    eliminate=True,
    analyze_topology=True,
    value_symbol=None,
    topology_options=None,
):
    """Analyze exact parameter-dependent values, optimizer loci and fiber changes.

    CAD topology is owned by semialg. One-dimensional fibers provide Hardt
    graph/band strata; higher-dimensional fibers provide complete cylindrical
    cells and dimensions. General Betti numbers and cross-stratum homeomorphisms
    are not inferred from cell counts.
    """
    if sense not in {"minimize", "maximize"}:
        raise ValueError("sense must be minimize or maximize")
    objective, domain, coordinates, prepared = _problem(
        objective, constraints, variables, region, bounds
    )
    params = tuple(normalize_variables(parameters))
    if set(params).intersection(coordinates) or (
        objective.free_symbols | domain.free_symbols
    ) - {*params, *coordinates}:
        raise ValueError(
            "declare distinct decision variables and every coefficient parameter"
        )
    value = sp.Dummy("value", real=True) if value_symbol is None else value_symbol
    if (
        not isinstance(value, sp.Symbol)
        or value in (*params, *coordinates)
        or value.is_real is False
    ):
        raise ValueError("value_symbol must be a fresh real symbol")
    from semialg.quantifiers import Exists, ForAll

    other = tuple(sp.Dummy("fiber", real=True) for _ in coordinates)
    rename = dict(zip(coordinates, other))
    feasible = Exists(other, domain.xreplace(rename))
    lower = ForAll(
        other,
        sp.Or(
            sp.Not(domain.xreplace(rename)),
            objective.xreplace(rename) >= value
            if sense == "minimize"
            else objective.xreplace(rename) <= value,
        ),
    )
    epsilon = sp.Dummy("epsilon", real=True)
    tight = ForAll(
        (epsilon,),
        sp.Or(
            epsilon <= 0,
            Exists(
                other,
                sp.And(
                    domain.xreplace(rename),
                    objective.xreplace(rename) < value + epsilon
                    if sense == "minimize"
                    else objective.xreplace(rename) > value - epsilon,
                ),
            ),
        ),
    )
    relation = sp.And(lower, tight)
    optimizer = optimizer_locus(
        objective, domain, variables=coordinates, sense=sense, eliminate=False
    )
    attained = Exists(coordinates, optimizer)
    threshold = sp.Dummy("threshold", real=True)
    unbounded = ForAll(
        (threshold,),
        Exists(
            other,
            sp.And(
                domain.xreplace(rename),
                objective.xreplace(rename) < threshold
                if sense == "minimize"
                else objective.xreplace(rename) > threshold,
            ),
        ),
    )
    if eliminate:
        relation = _eliminate_family(relation, (*params, value), params, prepared)
        optimizer = _eliminate_family(
            optimizer, (*params, *coordinates), params, prepared
        )
        attained = _eliminate_family(attained, params, params, prepared)
        feasible = _eliminate_family(feasible, params, params, prepared)
        unbounded = _eliminate_family(unbounded, params, params, prepared)
    topology = None
    if analyze_topology:
        from semialg.region_families import region_family

        source = (
            prepared.source
            if prepared is not None
            and prepared.is_geometry
            and constraints is None
            and bounds is None
            and objective.is_polynomial(*coordinates)
            else domain
        )
        topology = region_family(
            source, coordinates, params, **(topology_options or {})
        )
    return OptimizationFamily(
        objective,
        coordinates,
        params,
        value,
        relation,
        optimizer,
        attained,
        feasible,
        unbounded,
        topology,
        eliminate,
    )
