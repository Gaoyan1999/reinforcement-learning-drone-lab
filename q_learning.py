"""A small, explicit implementation of tabular Q-learning."""

from __future__ import annotations

import random

import numpy as np

from env import Position


class QLearningAgent:
    """Learn an action-value table for a grid environment.

    ``q_values[x, y, action]`` estimates the future discounted reward earned
    after choosing ``action`` from grid cell ``(x, y)``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        action_count: int,
        learning_rate: float = 0.2,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.action_count = action_count
        # One Q-value for every (x, y, action) combination; zero means no experience yet.
        # Example: q_values[2, 3, 1] is the value of moving right from cell (2, 3).
        self.q_values = np.zeros((width, height, action_count), dtype=float)

    def choose_action(
        self,
        state: Position,
        rng: random.Random,
        explore: bool = True,
    ) -> int:
        """Choose randomly with probability epsilon; otherwise choose a best action."""
        if explore and rng.random() < self.epsilon:
            return rng.randrange(self.action_count)

        values = self.q_values[state[0], state[1]]
        best_actions = np.flatnonzero(values == values.max())
        return int(rng.choice(best_actions))

    def update(
        self,
        state: Position,
        action: int,
        reward: float,
        next_state: Position,
        done: bool,
    ) -> float:
        """Update one table entry and return its temporal-difference error.

        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        The future-value term is zero if this transition ends the episode.
        """
        current_value = self.q_values[state[0], state[1], action]
        next_best_value = 0.0 if done else self.q_values[next_state[0], next_state[1]].max()
        target = reward + self.discount_factor * next_best_value
        td_error = target - current_value
        self.q_values[state[0], state[1], action] += self.learning_rate * td_error
        return float(td_error)
