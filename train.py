"""Train and evaluate a tabular Q-learning agent in the grid drone environment."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from env import GridDroneEnv
from q_learning import QLearningAgent


@dataclass
class TrainingHistory:
    returns: list[float]
    steps: list[int]
    successes: list[bool]


def train(
    env: GridDroneEnv,
    agent: QLearningAgent,
    episodes: int,
    rng: random.Random,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99,
) -> TrainingHistory:
    """Train the agent and slowly reduce random exploration."""
    history = TrainingHistory(returns=[], steps=[], successes=[])

    for episode in range(episodes):
        agent.epsilon = max(epsilon_end, epsilon_start * (epsilon_decay**episode))
        state = env.reset()
        episode_return = 0.0

        while True:
            action = agent.choose_action(state, rng)
            result = env.step(action)
            agent.update(state, action, result.reward, result.state, result.done)
            state = result.state
            episode_return += result.reward

            if result.done:
                history.returns.append(episode_return)
                history.steps.append(result.info["steps"])
                history.successes.append(result.info["reached_target"])
                break

    return history


def evaluate(
    env: GridDroneEnv,
    agent: QLearningAgent,
    episodes: int,
    rng: random.Random,
) -> float:
    """Measure the success rate using the learned greedy policy only."""
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = agent.choose_action(state, rng, explore=False)
            result = env.step(action)
            state = result.state
            if result.done:
                successes += int(result.info["reached_target"])
                break
    return successes / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")

    env = GridDroneEnv()
    agent = QLearningAgent(env.width, env.height, len(env.ACTIONS))
    rng = random.Random(args.seed)
    history = train(env, agent, args.episodes, rng)
    success_rate = evaluate(env, agent, episodes=100, rng=rng)

    print(f"Training episodes: {args.episodes}")
    print(f"Final training epsilon: {agent.epsilon:.3f}")
    print(f"Final-50 training success rate: {sum(history.successes[-50:]) / min(50, args.episodes):.1%}")
    print(f"Greedy evaluation success rate: {success_rate:.1%}")


if __name__ == "__main__":
    main()
