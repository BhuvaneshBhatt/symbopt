# Documentation maintenance

The README is the project landing page. The Sphinx/MyST tree under `docs/` is the
authoritative manual, while `notebooks/symbopt_tutorial.ipynb` is the executable
long-form tutorial.

## Local build

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Read the Docs

`.readthedocs.yaml` installs the package with the `docs` optional dependency and
builds `docs/conf.py`.

The package metadata should point `Documentation` to the hosted Read the Docs
site only after that project is created and successfully building. Until then,
the repository `docs/` URL is the truthful documentation URL.

## Public API coverage

`tests/test_documentation.py` checks that every stable name exported by
`symbopt.api.__all__` appears in the API reference source. This keeps new public
entry points from being added without documentation.
