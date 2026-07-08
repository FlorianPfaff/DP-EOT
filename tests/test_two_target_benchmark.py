from dpeot.experiments.run_two_target_crossing import run_benchmark


def test_two_target_benchmark_returns_all_methods() -> None:
    rows = run_benchmark(num_trials=1, base_seed=13)
    methods = {row["method"] for row in rows}

    assert methods == {
        "distance_collapse",
        "dp_x_order",
        "proposed_group_labels",
        "oracle_identity",
    }
    for row in rows:
        assert float(row["runtime_ms_per_scan"]) >= 0.0
