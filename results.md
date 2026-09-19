# Experiment Results

## Setup

All experiments use the obstacle target-detection task and the same random
seed (`7`). Each configuration trains for 1,000 episodes, then runs a separate
100-episode greedy evaluation with exploration disabled.

## Results

| Configuration | Learning rate (alpha) | Discount factor (gamma) | Final 100 training episodes | Greedy evaluation |
|---|---:|---:|---:|---:|
| Effective default | 0.20 | 0.95 | 100% | 100% |
| Short-sighted | 0.20 | 0.00 | 48% | 50% |
| Slow learning | 0.01 | 0.95 | 100% | 100% |

## Interpretation

### Effective default configuration

With `alpha = 0.20` and `gamma = 0.95`, the agent reliably learns to route
around the obstacle and enter the target-detection radius. The non-zero
discount factor lets the terminal `+10` reward propagate backward through the
earlier states on a successful trajectory.

### Failed configuration: gamma = 0

With `gamma = 0`, an action is valued only by its immediate reward. Normal
moves always receive `-0.1`, so states before the target cannot benefit from
the target's future `+10` reward. The agent still succeeds sometimes through
remaining random exploration and tie-breaking, but the 50% greedy evaluation
shows that it did not learn a reliable route.

### Slow learning rate

With `alpha = 0.01`, each update is much smaller. This simple, deterministic
map still allows the agent to converge within 1,000 episodes, but a harder or
more stochastic task would normally make this setting learn more slowly.

## Reproduce

```bash
python3 experiments.py
```

The command writes the raw CSV values and a comparison chart to
`artifacts/experiments/`. These local artifacts are intentionally excluded
from Git.
