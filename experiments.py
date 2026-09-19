"""Compare Q-learning hyperparameters on the obstacle target-detection task."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from detection_task import build_detection_environment
from q_learning import QLearningAgent
from train import evaluate, train


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    learning_rate: float
    discount_factor: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    final_training_success_rate: float
    greedy_evaluation_success_rate: float


CONFIGS = (
    ExperimentConfig(
        name="effective_default",
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
    ),
    ExperimentConfig(
        name="short_sighted_gamma_0",
        learning_rate=0.2,
        discount_factor=0.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
    ),
    ExperimentConfig(
        name="slow_learning_rate",
        learning_rate=0.01,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
    ),
)


def run_experiment(config: ExperimentConfig, episodes: int, seed: int) -> ExperimentResult:
    """Train one fresh agent and return training and greedy-evaluation metrics."""
    environment = build_detection_environment()
    agent = QLearningAgent(
        environment.width,
        environment.height,
        len(environment.ACTIONS),
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
    )
    rng = random.Random(seed)
    history = train(
        environment,
        agent,
        episodes,
        rng,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
    )
    final_window = min(100, episodes)
    return ExperimentResult(
        config=config,
        final_training_success_rate=sum(history.successes[-final_window:]) / final_window,
        greedy_evaluation_success_rate=evaluate(environment, agent, episodes=100, rng=rng),
    )


def save_results(results: list[ExperimentResult], output_directory: Path) -> None:
    """Write a machine-readable summary and a compact visual comparison."""
    csv_path = output_directory / "summary.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "name",
                "learning_rate",
                "discount_factor",
                "final_training_success_rate",
                "greedy_evaluation_success_rate",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.config.name,
                    "learning_rate": result.config.learning_rate,
                    "discount_factor": result.config.discount_factor,
                    "final_training_success_rate": result.final_training_success_rate,
                    "greedy_evaluation_success_rate": result.greedy_evaluation_success_rate,
                }
            )

    names = [result.config.name.replace("_", "\n") for result in results]
    evaluation_rates = [result.greedy_evaluation_success_rate for result in results]
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, evaluation_rates, color=("#16a34a", "#dc2626", "#2563eb"))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Greedy evaluation success rate")
    axis.set_title("Hyperparameter comparison after training")
    axis.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, evaluation_rates):
        axis.text(bar.get_x() + bar.get_width() / 2, rate + 0.03, f"{rate:.0%}", ha="center")
    figure.tight_layout()
    figure.savefig(output_directory / "comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1_000, help="training episodes per configuration")
    parser.add_argument("--seed", type=int, default=7, help="random seed for each configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="directory for CSV and PNG experiment results",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_experiment(config, args.episodes, args.seed) for config in CONFIGS]
    save_results(results, args.output_dir)

    for result in results:
        print(
            f"{result.config.name}: "
            f"final training={result.final_training_success_rate:.1%}, "
            f"greedy evaluation={result.greedy_evaluation_success_rate:.1%}"
        )
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
