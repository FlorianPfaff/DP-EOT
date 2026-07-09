from dpeot.scenarios.two_target_merge_split import (
    ScenarioConfig,
    generate_two_target_merge_split,
    oracle_cells,
)


def test_two_target_scenario_marks_unresolved_interval() -> None:
    config = ScenarioConfig(num_steps=12, merge_start=4, merge_end=6, seed=1)
    scenario = generate_two_target_merge_split(config)

    assert scenario.labels == ("A", "B")
    assert len(scenario.scans) == config.num_steps
    assert scenario.scans[3].unresolved_members == frozenset()
    assert scenario.scans[4].unresolved_members == frozenset({"A", "B"})
    assert scenario.scans[6].unresolved_members == frozenset({"A", "B"})
    assert scenario.scans[7].unresolved_members == frozenset()


def test_oracle_group_partition_combines_members_during_merge() -> None:
    config = ScenarioConfig(num_steps=12, merge_start=4, merge_end=6, seed=2)
    scenario = generate_two_target_merge_split(config)
    scan = scenario.scans[4]

    grouped = oracle_cells(scan, group_during_unresolved=True)
    separated = oracle_cells(scan, group_during_unresolved=False)

    assert len(grouped) <= len(separated)
    assert any(len(cell) > 1 for cell in grouped)


def test_scenario_supports_asymmetric_rates_extents_and_crossing_angle() -> None:
    config = ScenarioConfig(
        measurement_rates=(16.0, 8.0),
        extent_axes=(0.8, 0.25),
        extent_axes_b=(1.2, 0.2),
        crossing_y_offset=2.0,
        seed=3,
    )
    scenario = generate_two_target_merge_split(config)
    target_a, target_b = scenario.targets

    assert target_a.measurement_rate == 16.0
    assert target_b.measurement_rate == 8.0
    assert target_a.extent[0, 0] != target_b.extent[0, 0]
    assert target_a.states[0, 1] == -2.0
    assert target_b.states[0, 1] == 2.0
