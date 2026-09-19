import unittest

from env import GridDroneEnv
from q_learning import QLearningAgent


class GridDroneEnvTests(unittest.TestCase):
    def test_reaching_target_ends_episode(self) -> None:
        env = GridDroneEnv(width=3, height=2, start=(0, 0), target=(1, 0))

        result = env.step(1)

        self.assertEqual(result.state, (1, 0))
        self.assertEqual(result.reward, 10.0)
        self.assertTrue(result.done)
        self.assertTrue(result.info["reached_target"])

    def test_boundary_collision_keeps_position(self) -> None:
        env = GridDroneEnv(width=3, height=3, start=(0, 0), target=(2, 2))

        result = env.step(3)

        self.assertEqual(result.state, (0, 0))
        self.assertEqual(result.reward, -1.0)
        self.assertTrue(result.info["collision"])

    def test_obstacle_collision_keeps_position(self) -> None:
        env = GridDroneEnv(
            width=3,
            height=3,
            start=(0, 0),
            target=(2, 2),
            obstacles={(1, 0)},
        )

        result = env.step(1)

        self.assertEqual(result.state, (0, 0))
        self.assertEqual(result.reward, -1.0)
        self.assertTrue(result.info["collision"])

    def test_episode_times_out_at_maximum_steps(self) -> None:
        env = GridDroneEnv(width=3, height=3, start=(0, 0), target=(2, 2), max_steps=2)

        env.step(1)
        result = env.step(1)

        self.assertTrue(result.done)
        self.assertTrue(result.info["timed_out"])

    def test_detection_radius_ends_episode_before_reaching_target_cell(self) -> None:
        env = GridDroneEnv(
            width=5,
            height=3,
            start=(0, 0),
            target=(3, 1),
            detection_radius=1.0,
        )
        env.step(1)
        env.step(1)

        result = env.step(0)

        self.assertEqual(result.state, (2, 1))
        self.assertNotEqual(result.state, env.target)
        self.assertTrue(result.done)
        self.assertTrue(result.info["detected_target"])
        self.assertEqual(result.reward, 10.0)

    def test_reset_restores_initial_state(self) -> None:
        env = GridDroneEnv(width=3, height=3, start=(0, 0), target=(2, 2))
        env.step(1)

        state = env.reset()

        self.assertEqual(state, (0, 0))
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.trajectory, [(0, 0)])

    def test_q_learning_terminal_update_uses_reward_only(self) -> None:
        agent = QLearningAgent(
            width=2,
            height=2,
            action_count=4,
            learning_rate=0.5,
            discount_factor=0.95,
            epsilon=0.0,
        )

        td_error = agent.update((0, 0), action=1, reward=10.0, next_state=(1, 0), done=True)

        self.assertEqual(td_error, 10.0)
        self.assertEqual(agent.q_values[0, 0, 1], 5.0)

    def test_q_learning_non_terminal_update_includes_future_value(self) -> None:
        agent = QLearningAgent(
            width=2,
            height=2,
            action_count=4,
            learning_rate=1.0,
            discount_factor=0.5,
            epsilon=0.0,
        )
        agent.q_values[1, 0, 3] = 8.0

        agent.update((0, 0), action=1, reward=1.0, next_state=(1, 0), done=False)

        self.assertEqual(agent.q_values[0, 0, 1], 5.0)


if __name__ == "__main__":
    unittest.main()
