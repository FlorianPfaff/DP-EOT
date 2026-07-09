from dpeot.experiments.run_two_target_crossing import run_benchmark


def test_two_target_benchmark_returns_all_methods() -> None:
    rows = run_benchmark(num_trials=1, base_seed=13)
    methods = {row["method"] for row in rows}

    assert methods == {
        "distance_collapse",
        "dp_x_order",
        "mfm_x_order",
        "proposed_group_labels",
        "oracle_identity",
    }
    for row in rows:
        assert "id_switches_total" in row
        assert "id_switches_pre_merge" in row
        assert "id_switches_during_unresolved" in row
        assert "id_switches_post_split" in row
        assert "label_recovery_post_split" in row
        assert "split_recovery_delay" in row
        assert "group_membership_during_unresolved" in row
        assert row["id_switches"] == row["id_switches_total"]
        assert row["label_recovery"] == row["label_recovery_post_split"]
        assert float(row["runtime_ms_per_scan"]) >= 0.0
