# Installation

## Requirements

`symbopt` supports CPython 3.11 through 3.14. Its core dependencies are SymPy,
NumPy, SciPy, and `python-flint`, which supplies the Arb real-ball backend used
for rigorous transcendental interval enclosures.

Install from PyPI once a release is available:

```bash
python -m pip install symbopt
```

For development from a clone:

```bash
git clone https://github.com/BhuvaneshBhatt/symbopt.git
cd symbopt
python -m pip install -e ".[test,docs,notebook,release]"
```

## Verify the installation

```python
import symbopt

print(symbopt.arb_available())
```

A normal release installation should print `True`. Arb is required for rigorous
verified enclosures of supported transcendental expressions. Exact polynomial
Bernstein bounds do not require Arb, but the release CI requires the Arb backend
so that transcendental verification tests cannot silently skip.

A minimal optimization check is:

```python
import sympy as sp
from symbopt import minimize

x = sp.symbols("x", real=True)
result = minimize((x - 2)**2, [sp.Ge(x, 0)])
print(result.optimum_value)
```

The expected minimum value is `0`.

## Documentation build

The documentation uses Sphinx with MyST Markdown:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The repository also contains `.readthedocs.yaml` for hosted builds.


## Optional CAD/QE backend

Install `semialg` support with:

```bash
python -m pip install "symbopt[semialg]"
```

The backend is imported lazily. Without it, LP/QP, symbolic/KKT, SOS/moment,
and verified-interval functionality remains available.
