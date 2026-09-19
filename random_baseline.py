"""Run and render a random-policy baseline for the grid drone environment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from env import GridDroneEnv


def run_episode(env: GridDroneEnv, rng: random.Random) -> tuple[float, bool]:
    """Run one random episode and return its total reward and success flag."""
    env.reset()
    total_reward = 0.0

    while True:
        result = env.step(rng.choice(tuple(env.ACTIONS)))
        total_reward += result.reward
        if result.done:
            return total_reward, result.info["reached_target"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="number of random episodes")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--render-path",
        type=Path,
        default=Path("artifacts/random_baseline.png"),
        help="where to save the final trajectory image",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")

    env = GridDroneEnv()
    rng = random.Random(args.seed)
    successes = 0
    total_return = 0.0

    for _ in range(args.episodes):
        episode_return, reached_target = run_episode(env, rng)
        total_return += episode_return
        successes += int(reached_target)

    args.render_path.parent.mkdir(parents=True, exist_ok=True)
    env.render(str(args.render_path))

    print(f"Episodes: {args.episodes}")
    print(f"Random-policy success rate: {successes / args.episodes:.1%}")
    print(f"Average return: {total_return / args.episodes:.2f}")
    print(f"Final trajectory saved to: {args.render_path}")


if __name__ == "__main__":
    main()
