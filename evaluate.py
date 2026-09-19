"""Train a Q-learning agent and save visual evidence of its behaviour."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env import GridDroneEnv
from q_learning import QLearningAgent
from train import TrainingHistory, evaluate, train


def run_random_episode(env: GridDroneEnv, rng: random.Random) -> bool:
    """Run one random episode and leave its trajectory on the environment."""
    env.reset()
    while True:
        result = env.step(rng.randrange(len(env.ACTIONS)))
        if result.done:
            return bool(result.info["reached_target"])


def run_greedy_episode(
    env: GridDroneEnv,
    agent: QLearningAgent,
    rng: random.Random,
) -> bool:
    """Run one episode using only the learned best actions."""
    state = env.reset()
    while True:
        action = agent.choose_action(state, rng, explore=False)
        result = env.step(action)
        state = result.state
        if result.done:
            return bool(result.info["reached_target"])


def rolling_mean(values: list[float] | list[bool], window: int = 25) -> np.ndarray:
    """Return a trailing average, using the available values at the beginning."""
    if window < 1:
        raise ValueError("window must be at least 1")
    array = np.asarray(values, dtype=float)
    cumulative_sum = np.cumsum(np.insert(array, 0, 0.0))
    start_indices = np.maximum(0, np.arange(len(array)) - window + 1)
    counts = np.arange(1, len(array) + 1) - start_indices
    return (cumulative_sum[1:] - cumulative_sum[start_indices]) / counts


def plot_training_history(history: TrainingHistory, save_path: Path) -> None:
    """Save reward, success-rate, and episode-length curves."""
    episodes = np.arange(1, len(history.returns) + 1)
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(episodes, rolling_mean(history.returns), color="#2563eb")
    axes[0].set_ylabel("Mean return")
    axes[0].set_title("Q-learning training progress (25-episode trailing mean)")

    axes[1].plot(episodes, rolling_mean(history.successes), color="#16a34a")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(episodes, rolling_mean(history.steps), color="#9333ea")
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("Mean steps")

    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="training episodes")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="directory for the generated PNG files",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    env = GridDroneEnv()

    random_success = run_random_episode(env, rng)
    env.render(str(args.output_dir / "random_trajectory.png"))

    agent = QLearningAgent(env.width, env.height, len(env.ACTIONS))
    history = train(env, agent, args.episodes, rng)
    plot_training_history(history, args.output_dir / "training_curves.png")

    greedy_success = run_greedy_episode(env, agent, rng)
    env.render(str(args.output_dir / "greedy_trajectory.png"))
    success_rate = evaluate(env, agent, episodes=100, rng=rng)

    print(f"Random trajectory reached target: {random_success}")
    print(f"Greedy trajectory reached target: {greedy_success}")
    print(f"Greedy 100-episode success rate: {success_rate:.1%}")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
