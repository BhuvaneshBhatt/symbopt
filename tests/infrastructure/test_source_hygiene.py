def test_lifted_problem_does_not_store_unused_constraint_kind_metadata():
    from symbopt.problem_models import LiftedProblem

    assert "constraint_kinds" not in LiftedProblem.__dataclass_fields__
