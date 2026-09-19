# Reinforcement Learning Drone Lab

A small, from-scratch reinforcement-learning project for learning how an agent can navigate a 2D map, avoid obstacles, and detect a target.

The first version uses tabular Q-learning, NumPy, and Matplotlib only. It deliberately avoids heavyweight simulators and reinforcement-learning frameworks so that every important part of the training loop remains visible.

## Learning outcomes

- Design an RL environment with `reset()` and `step(action)`.
- Train and evaluate a tabular Q-learning agent.
- Visualise trajectories and learning curves.
- Experiment with exploration and reward design.
- Extend a navigation task into a target-detection task.

## Roadmap

See [plan.md](plan.md) for the implementation plan and completion criteria.

## Run the first baseline

Install the small set of dependencies and run a random policy:

```bash
python3 -m pip install numpy matplotlib
python3 random_baseline.py
```

The command reports the random-policy success rate and writes the final flight
trajectory to `artifacts/random_baseline.png`.

Run the environment checks with:

```bash
python3 -m unittest discover -s tests
```

## Train the Q-learning agent

After the baseline, train the tabular Q-learning agent:

```bash
python3 train.py
```

The script trains for 500 episodes and reports both the final training success
rate and a 100-episode greedy evaluation success rate.

## Visualise the training result

Create learning curves and compare one random trajectory with one learned,
greedy trajectory:

```bash
python3 evaluate.py
```

The command writes three PNG files to `artifacts/evaluation/`:

- `training_curves.png`: trailing mean return, success rate, and episode length;
- `random_trajectory.png`: one untrained random flight;
- `greedy_trajectory.png`: one flight using the learned Q-table.

## Status

Stage 2 in progress: a tabular Q-learning agent and training script are available.
