import unittest

from env import GridDroneEnv


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

    def test_episode_times_out_at_maximum_steps(self) -> None:
        env = GridDroneEnv(width=3, height=3, start=(0, 0), target=(2, 2), max_steps=2)

        env.step(1)
        result = env.step(1)

        self.assertTrue(result.done)
        self.assertTrue(result.info["timed_out"])

    def test_reset_restores_initial_state(self) -> None:
        env = GridDroneEnv(width=3, height=3, start=(0, 0), target=(2, 2))
        env.step(1)

        state = env.reset()

        self.assertEqual(state, (0, 0))
        self.assertEqual(env.steps, 0)
        self.assertEqual(env.trajectory, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
