"""Train Q-learning on a drone target-detection task with obstacles."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from env import GridDroneEnv
from evaluate import plot_training_history, run_greedy_episode
from q_learning import QLearningAgent
from train import evaluate, train


def build_detection_environment() -> GridDroneEnv:
    """Create a small obstacle course with a one-cell target-detection radius."""
    return GridDroneEnv(
        width=8,
        height=8,
        start=(0, 0),
        # The target sits behind the vertical wall, so the drone must fully route around it.
        target=(6, 4),
        obstacles={(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (4, 5)},
        max_steps=80,
        detection_radius=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1_000, help="training episodes")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/detection_task"),
        help="directory for the generated PNG files",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    env = build_detection_environment()
    agent = QLearningAgent(env.width, env.height, len(env.ACTIONS))

    history = train(env, agent, args.episodes, rng)
    plot_training_history(history, args.output_dir / "training_curves.png")

    success = run_greedy_episode(env, agent, rng)
    env.render(str(args.output_dir / "greedy_trajectory.png"))
    success_rate = evaluate(env, agent, episodes=100, rng=rng)

    print(f"Greedy trajectory detected target: {success}")
    print(f"Greedy 100-episode success rate: {success_rate:.1%}")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
