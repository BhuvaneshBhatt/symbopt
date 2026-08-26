# Contributing

Contributions are welcome through the GitHub repository:

https://github.com/BhuvaneshBhatt/symbopt

## Development setup

```bash
git clone https://github.com/BhuvaneshBhatt/symbopt.git
cd symbopt
python -m pip install -e ".[test,docs,notebook]"
```

Run tests before opening a pull request:

```bash
python -m pytest -q
```

Build documentation with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Code expectations

Contributions should preserve the package's mathematical semantics and current
style:

- use descriptive names,
- avoid compatibility aliases unless there is a strong release reason,
- do not add implementation-history terminology to public names/docs,
- centralize repeated mathematical conventions,
- add comments for non-obvious mathematics rather than narrating obvious code,
- avoid trailing whitespace,
- include regression tests for bug fixes,
- prefer an inconclusive certificate over an unjustified one.

## Mathematical changes

A pull request that changes a sign convention, constraint normalization,
certificate rule, interval-rigorousness rule, or SDP/SOS identity should include
a small mathematical example demonstrating the intended semantics.

## Documentation changes

Update the relevant conceptual/method page as well as API documentation when a
public behavior changes.
