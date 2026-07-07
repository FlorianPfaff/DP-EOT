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
