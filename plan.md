# Reinforcement Learning Drone Lab — Practice Plan

## Project goal

Build and train a two-dimensional drone reinforcement-learning environment from scratch. The drone must navigate within a limited number of steps, avoid obstacles, and detect a target. The emphasis is on transparent environment code, training loops, evaluation, and experimentation rather than opaque frameworks.

## Technology scope

- Python 3
- NumPy for Q-tables and numerical operations
- Matplotlib for learning curves and trajectory visualisation
- No Gymnasium, PyTorch, GPU, or heavyweight flight simulator in the first version

## Implementation roadmap

### 1. Environment and random baseline

- Create `env.py` with `reset()` and `step(action)` methods.
- Define a 2D grid, boundaries, a drone, and a target.
- Implement four actions: up, down, left, and right.
- Run a random policy to verify state transitions, rewards, and episode termination.

**Done when:** a random agent can complete an episode and render its trajectory.

### 2. Tabular Q-learning

- Create `q_learning.py` to maintain `Q[state, action]`.
- Implement epsilon-greedy exploration and the Q-learning update.
- Train over many episodes while storing return, steps, and success status.

**Done when:** on a fixed-target, obstacle-free map, the greedy policy clearly outperforms the random baseline.

### 3. Evaluation and visualisation

- Create `evaluate.py` to keep training and evaluation separate.
- Plot episode returns, success rate, and episode length.
- Render trajectories before and after training rather than relying only on a single reward number.

**Done when:** an independent evaluation reaches at least an 85% success rate.

### 4. Target-detection task

- Add a target-detection radius: reaching the radius detects the target and ends the episode.
- Add obstacles, a maximum step count, and a per-step cost.
- Compare reward designs: success reward, step penalty, collision penalty, and distance shaping.

**Done when:** the drone finds the target from several starting positions without taking obviously wasteful routes.

### 5. Experiments and reflection

- Compare exploration rate, learning rate, and discount factor settings.
- Record one failed and one effective configuration, and explain how rewards and exploration affected learning.
- Summarise observations and future work in `results.md`.

## Optional extensions

1. Random target positions and limited sensor range.
2. Replace the Q-table with a DQN for larger state spaces.
3. Add a moving target or more realistic flight dynamics.

## Planned structure

```text
reinforcement-learning-drone-lab/
├── README.md
├── plan.md
├── env.py
├── q_learning.py
├── train.py
├── evaluate.py
├── render.py
├── results.md
└── tests/
```
