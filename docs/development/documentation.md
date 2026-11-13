# Documentation maintenance

The README is the project landing page. The Sphinx/MyST tree under `docs/` is the authoritative manual, while `notebooks/symbopt_tutorial.ipynb` is the executable long-form tutorial.

## Local build

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Read the Docs

`.readthedocs.yaml` installs the package with the `docs` optional dependency and builds `docs/conf.py`.

The package metadata should point `Documentation` to the hosted Read the Docs site only after that project is created and successfully building. Until then, the repository `docs/` URL is the truthful documentation URL.

## Public API coverage

`tests/tooling/test_documentation.py` checks that every stable name exported by `symbopt.api.__all__` appears in the API reference source. This keeps new public entry points from being added without documentation.

## Capability examples

Both tables in {doc}`../reference/capability-matrix` link every data row to a section of {doc}`../examples/capability-tour`. `test_capability_examples.py` resolves each local anchor and requires an executable Python assertion in that section. `test_mathematical_examples.py` executes the whole page in a fresh namespace from both source and installed-wheel harnesses. Link tests do not depend on the wording or order of the capability labels.

An unsupported row should link to a checked refusal or a mathematical boundary example. Optional examples state their required backend, run when it is installed, and are covered by the optional-analysis wheel job. Do not make the capability matrix promise a general algorithm from a successful special case.
