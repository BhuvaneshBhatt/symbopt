# Optimization over regions

Use `region=` when the feasible domain is already represented as geometry, a symbolic region or a SymPy set. Symbopt can reuse structural bounds, reduce a covering parameterization to fewer coordinates and share rigorous bounds across disconnected branches. A successful conversion still does not guarantee that the chosen optimization algorithm completes.

## Sets and formulas

Core installations accept Boolean formulas, intervals, scalar finite sets, Cartesian products and finite sets of coordinate tuples. Supply `variables=` in coordinate order. Unresolved membership predicates, such as general `ImageSet` membership, raise `NotImplementedError`.

```python
import sympy as sp
import symbopt as so

x, y, t, a, value = sp.symbols("x y t a value", real=True)
assert so.minimize(-x, region=sp.Interval(0, 1), variables=x) == [-1, {x: 1}]
rectangle = sp.ProductSet(sp.Interval(0, 1), sp.Interval(2, 3))
assert so.minimize(x + y, region=rectangle, variables=(x, y)) == [2, {x: 0, y: 2}]
points = sp.FiniteSet(sp.Tuple(1, 2), sp.Tuple(-2, 3))
assert so.minimize(x * x + y * y, region=points, variables=(x, y)) == [5, {x: 1, y: 2}]
```

The global minimum/maximum wrappers, their detailed-result variants, value helpers and optimizer helpers share this interface. Local optimization entry points do not accept `region=`. Geometry and CAD/QE remain owned by semialg; optimization scheduling and result lifting belong to symbopt.

## Prepared regions and bounds

`prepare_region` retains coordinate ownership and caches exact membership, structural charts and outer bounds independently. Preparing does not immediately request quantifier elimination. Reuse the object across objectives.

```python
interval = so.prepare_region(sp.Interval.open(0, 1), x)
assert interval.outer_bounds == {x: (0, 1)}
assert so.minval(x, region=interval) == 0
assert so.maxval(x, region=interval) == 1
assert interval.provenance["membership_ready"]
```

Install `symbopt[semialg]` for geometry objects, symbolic/CAD regions and parameter analysis. Structural bounds cover boxes, balls, ellipsoids, vertices, unions and supported polynomial images without calling a coordinate optimizer. They may be loose. Numeric finite covers feed interval search; symbolic bounds remain symbolic until specialized. Original membership always defines feasibility.

```python
if so.semialg_available():
    from semialg import Ball, Box, ParametricRegion, SemialgebraicRegion, Simplex

    ball = so.prepare_region(Ball((1, 2), 3), (x, y))
    assert ball.outer_bounds == {x: (-2, 4), y: (-1, 5)}
    assert so.minval(x + y, region=ball) == 3 - 3 * sp.sqrt(2)
```

Sets, formulas and coordinate-free geometry require explicit decision variables. Symbolic/CAD regions supply their owned coordinate order; an explicit order must match. All other symbols remain coefficients or parameters.

Conversion has deterministic admission controls: `max_operations` and `max_quantifiers`, with defaults 5000 and 12. `None` disables a limit. `RegionConversionError.stage` and `.reason` identify a refusal. These controls check the input before backend work; they are not wall-clock deadlines or limits on intermediate CAD growth. A structural pullback can solve without membership QE, even when quantified membership is prohibited by the conversion policy.

## Covering parameterizations

Complete structural charts support embedded simplices, supplied polynomial/rational images, transformed regions and their unions. Symbopt pulls the objective and every additional constraint into each chart, solves there and maps witnesses back. The chart need not be injective or regular. Point strata and boundaries remain present. No Jacobian or multiplicity factor is used for minimization.

```python
if so.semialg_available():
    curve = ParametricRegion((t,), ((t, 0, 1),), (t, t * t))
    prepared = so.prepare_region(curve, (x, y), max_quantifiers=0)
    minimum = so.minimize_result(y - x, region=prepared)
    assert minimum.optimum_value == -sp.Rational(1, 4)
    assert minimum.best_candidate.original_variable_values == {
        x: sp.Rational(1, 2),
        y: sp.Rational(1, 4),
    }
    assert not prepared.provenance["membership_ready"]
    assert so.minval(y - x, x >= sp.Rational(3, 4), region=prepared) == -sp.Rational(
        3, 16
    )
```

Chart parameters are renamed before substitution. When an ambient coordinate also occurs as a coefficient in the geometry, structural reduction is declined and the explicit membership formula defines the problem. Real rational maps retain denominator exclusions even for coordinates absent from the objective. Maps outside the supported real polynomial/rational reduction rules use ambient membership conversion. Structural chart and vertex proofs require exact coefficients. Inexact data use the existing precision-aware pipeline. Trigonometric sphere charts are left to the algebraic shape rules. Affine objectives on closed simplices, polytopes and polygons use vertex evaluation when no additional constraints alter the feasible hull.

## Intersections and disconnected regions

Membership, relational constraints and `bounds` are intersected. `bounds` restricts the mathematical domain; `box_bounds` supplies numerical search bounds under the existing search-coverage requirements. Open boundaries and holes retain their meaning.

```python
assert so.minimize(
    -x, x >= 0, region=sp.Interval(-2, 2), bounds={x: (None, 1)}, variables=x
) == [-1, {x: 1}]
open_result = so.minimize_result(x, region=sp.Interval.open(0, 1), variables=x)
assert open_result.optimum_value == 0 and open_result.attained is False
assert so.argmin(x, region=sp.Interval.open(0, 1), variables=x) == ()
assert so.minval(x, region=sp.EmptySet, variables=x) == sp.oo

disconnected = sp.Union(sp.Interval(-2, -1), sp.Interval(1, 2))
assert so.argmin(x * x, region=disconnected, variables=x) == ({x: -1}, {x: 1})
```

Disjunctive branches share an exact attained incumbent. A branch is skipped only when a rigorous enclosure puts its entire objective strictly above that value. Equality of bounds does not prune tied minimizers. The combined proof retains the branch bounds. Domains without a suitable finite product enclosure continue through ordinary solving.

## Exact optimizer loci

`PipelineResult.optimizer_locus` is the feasible level set when an exact global value has been proved. It includes optimizer continua and disconnected sets beyond the represented candidate mappings. An unknown global value gives `None`; an empty domain, an unbounded problem or a proved unattained infimum gives `False`.

```python
if so.semialg_available():
    locus = so.optimizer_locus(x * x, region=Box(((-1, 1), (-2, 2))), variables=(x, y))
    assert sp.simplify(locus.subs({x: 0, y: 2})) is sp.true
    assert sp.simplify(locus.subs({x: 0, y: 3})) is sp.false
    assert sp.simplify(locus.subs({x: 1, y: 0})) is sp.false
```

The root `optimizer_locus` function first uses a completed value proof. Otherwise it defines all feasible points whose objective is no worse than every feasible competitor and requests real QE. With `eliminate=False`, it returns that exact quantified definition. A definition does not imply that QE completed or that a finite list of optimizer coordinates exists. General transcendental comparison formulas can be outside semialg's elimination scope.

## Parameter families

`optimization_family` keeps coefficient parameters separate from decision variables and returns a finite-value relation, the optimizer formula, feasibility, attainment and unboundedness conditions. Thus a finite unattained infimum differs from an empty fiber. `sense="maximize"` gives the corresponding supremum relations. The value symbol must be fresh. Coefficient sign assumptions are retained during elimination and fiber analysis.

```python
if so.semialg_available():
    family = so.optimization_family(
        x,
        region=SemialgebraicRegion(sp.And(x > 0, x < a), (x,)),
        parameters=(a,),
        value_symbol=value,
    )
    assert family.quantifier_free
    assert sp.simplify(family.value_relation.subs({a: 2, value: 0})) is sp.true
    assert sp.simplify(family.attained_condition.subs(a, 2)) is sp.false
    assert sp.simplify(family.feasible_condition.subs(a, 0)) is sp.false
    assert sp.simplify(family.unbounded_condition.subs(a, 2)) is sp.false
```

`eliminate=False` retains exact first-order definitions. `analyze_topology=False` skips a separate CAD analysis of the source fibers; `topology_options` forwards semialg's `max_cells` and `timeout` controls. Failed or resource-limited elimination raises; it never becomes an empty fiber or a claimed complete partition.

Semialg's `region_family` partitions parameter space into CAD cells and retains all selected fiber cells, including dimensions zero and one. For a single fiber coordinate it also supplies Hardt graph/band strata and exact component counts. In higher-dimensional fibers it supplies the full cylindrical decomposition and fiber dimension. Cell counts do not stand for connected-component counts, Betti numbers or a general homeomorphism classification. Polynomial ball/sphere families additionally have radial strata with exact component counts and Betti numbers across radius collapse. Automatic singular/nonlinear atlas construction and general higher-dimensional topological trivialization remain outside these interfaces.

## Quadratic certificates

`minimize_quadratic` and `verify_quadratic_certificate` accept region inputs. Replay must use the same original region and constraints. Native quadratic methods require one conjunction within their degree, coefficient and closed-boundary scope. Use the global wrappers for disjunctions and supported open-domain infima.

```python
rectangle = sp.ProductSet(sp.Interval(-1, 1), sp.Interval(-1, 1))
quadratic = so.minimize_quadratic(
    x * x - y * y, region=rectangle, variables=(x, y), backend="faces", certified=True
)
assert quadratic.objective_value == -1 and quadratic.certified
assert so.verify_quadratic_certificate(
    x * x - y * y, quadratic.certificate, region=rectangle, variables=(x, y)
)
```

These are exact methods within their stated mathematical and backend scope, with potentially substantial CAD/QE cost. See {doc}`../reference/limitations` and {doc}`../methods/global-quadratics` for the remaining solver restrictions.
