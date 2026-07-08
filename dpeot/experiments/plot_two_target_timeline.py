"""Create a diagnostic plot for the two-target merge/split scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dpeot.scenarios.two_target_merge_split import ScenarioConfig, generate_two_target_merge_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("two_target_merge_split.pdf"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    scenario = generate_two_target_merge_split(ScenarioConfig(seed=args.seed))
    plot_scenario(scenario, args.output)


def plot_scenario(scenario, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    for target in scenario.targets:
        states = target.states
        ax.plot(states[:, 0], states[:, 1], label=f"truth {target.label}")
        ax.scatter(states[0, 0], states[0, 1], marker="o")
        ax.scatter(states[-1, 0], states[-1, 1], marker="x")

    unresolved_measurements = [
        scan.measurements for scan in scenario.scans if scan.is_unresolved and scan.measurements.size
    ]
    resolved_measurements = [
        scan.measurements for scan in scenario.scans if not scan.is_unresolved and scan.measurements.size
    ]

    if resolved_measurements:
        points = np.vstack(resolved_measurements)
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.25, label="resolved scans")
    if unresolved_measurements:
        points = np.vstack(unresolved_measurements)
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5, label="unresolved scans")

    ax.set_title("Two-target merge/split scenario")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


if __name__ == "__main__":
    main()
