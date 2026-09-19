"""A minimal 2D grid environment for reinforcement-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


Position = tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    """The result of one environment transition."""

    state: Position
    reward: float
    done: bool
    info: Mapping[str, bool | int]


class GridDroneEnv:
    """Move a drone through a 2D grid until it reaches a target.

    Coordinates are represented as ``(x, y)``. The origin is at the bottom-left
    of the rendered map. Actions use the following integer encoding:

    - 0: up
    - 1: right
    - 2: down
    - 3: left

    A normal move costs ``step_penalty``. Attempting to leave the map or enter
    an obstacle keeps the drone in place and returns ``collision_penalty``.
    Reaching the target ends the episode with ``target_reward``.
    """

    ACTIONS: Mapping[int, Position] = {
        0: (0, 1),
        1: (1, 0),
        2: (0, -1),
        3: (-1, 0),
    }

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        start: Position = (0, 0),
        target: Position = (5, 5),
        obstacles: Iterable[Position] = (),
        max_steps: int = 50,
        step_penalty: float = -0.1,
        collision_penalty: float = -1.0,
        target_reward: float = 10.0,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("width and height must both be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        self.width = width
        self.height = height
        self.start = start
        self.target = target
        self.obstacles = frozenset(obstacles)
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.target_reward = target_reward

        for label, position in (("start", start), ("target", target)):
            self._validate_position(position, label)
        for obstacle in self.obstacles:
            self._validate_position(obstacle, "obstacle")
        if start == target:
            raise ValueError("start and target must be different positions")
        if start in self.obstacles or target in self.obstacles:
            raise ValueError("start and target cannot be obstacles")

        self.position: Position = self.start
        self.steps = 0
        self.trajectory: list[Position] = [self.position]

    def reset(self) -> Position:
        """Start a fresh episode and return the initial state."""
        self.position = self.start
        self.steps = 0
        self.trajectory = [self.position]
        return self.position

    def step(self, action: int) -> StepResult:
        """Apply one action and return the resulting transition."""
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action {action}; expected 0, 1, 2, or 3")

        dx, dy = self.ACTIONS[action]
        proposed_position = (self.position[0] + dx, self.position[1] + dy)
        collision = not self._is_valid_position(proposed_position)

        if collision:
            reward = self.collision_penalty
        else:
            self.position = proposed_position
            reward = self.step_penalty

        self.steps += 1
        self.trajectory.append(self.position)

        reached_target = self.position == self.target
        if reached_target:
            reward = self.target_reward

        timed_out = self.steps >= self.max_steps and not reached_target
        done = reached_target or timed_out
        return StepResult(
            state=self.position,
            reward=reward,
            done=done,
            info={
                "reached_target": reached_target,
                "collision": collision,
                "timed_out": timed_out,
                "steps": self.steps,
            },
        )

    def render(self, save_path: str | None = None):
        """Render the current episode; optionally write the figure to ``save_path``."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        figure, axis = plt.subplots(figsize=(6, 6))
        axis.set_xlim(-0.5, self.width - 0.5)
        axis.set_ylim(-0.5, self.height - 0.5)
        axis.set_xticks(range(self.width))
        axis.set_yticks(range(self.height))
        axis.grid(True, color="#d0d0d0", linewidth=1)
        axis.set_aspect("equal")
        axis.set_title("Drone trajectory")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        for obstacle_x, obstacle_y in self.obstacles:
            axis.add_patch(
                Rectangle(
                    (obstacle_x - 0.45, obstacle_y - 0.45),
                    0.9,
                    0.9,
                    color="#4b5563",
                    label="obstacle",
                )
            )

        axis.scatter(*self.start, s=150, marker="s", color="#2563eb", label="start", zorder=3)
        axis.scatter(*self.target, s=220, marker="*", color="#f59e0b", label="target", zorder=3)

        path_x, path_y = zip(*self.trajectory)
        axis.plot(path_x, path_y, color="#16a34a", linewidth=2, alpha=0.8, label="trajectory")
        axis.scatter(*self.position, s=110, color="#dc2626", label="drone", zorder=4)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        figure.tight_layout()

        if save_path is not None:
            figure.savefig(save_path, dpi=160, bbox_inches="tight")
        return figure

    def _is_valid_position(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height and position not in self.obstacles

    def _validate_position(self, position: Position, label: str) -> None:
        if not self._is_valid_position(position):
            raise ValueError(f"{label} position {position} is outside the map or blocked")
