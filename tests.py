import unittest
from io import StringIO
# for testing print output
from unittest.mock import patch

# gym default
import gymnasium as gym
from gymnasium import spaces

# gym tic tac toe
from gym_tictactoe.envs import TictactoeEnv
from gym_tictactoe.envs.helpers import *


class TestHelpers(unittest.TestCase):
    def test_base_x_to_dec(self):
        self.assertEqual(base_x_to_dec([1, 0], 2), 2)
        self.assertEqual(base_x_to_dec([1, 0, 0], 2), 4)
        self.assertEqual(base_x_to_dec([2, 0], 3), 6)

    def test_dec_to_base_x(self):
        self.assertEqual(dec_to_base_x(2, 2), [1, 0])
        self.assertEqual(dec_to_base_x(4, 2), [1, 0, 0])
        self.assertEqual(dec_to_base_x(6, 3), [2, 0])

    def test_list_to_matrix(self):
        self.assertEqual(list_to_matrix([0] * 9, 3), [[0] * 3] * 3)


class TestGym(unittest.TestCase):
    def test_creation(self):
        custom_win_reward = 1000
        custom_normal_reward = -2
        custom_violation_reward = -5
        custom_drawn_reward = -1
        custom_size = 4

        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1',
                                     reward_win=custom_win_reward,
                                     reward_normal=custom_normal_reward,
                                     reward_violation=custom_violation_reward,
                                     reward_drawn=custom_drawn_reward,
                                     size=custom_size).unwrapped
        env.reset()

        # check custom rewards
        self.assertEqual(env.reward_win, custom_win_reward)
        self.assertEqual(env.reward_normal, custom_normal_reward)
        self.assertEqual(env.reward_violation, custom_violation_reward)
        self.assertEqual(env.reward_drawn, custom_drawn_reward)

        # check action and observation space
        self.assertEqual(env.action_space, spaces.MultiDiscrete(
            [2, custom_size * custom_size]))
        self.assertEqual(env.observation_space, spaces.Discrete(
            3 ** (custom_size * custom_size)))

    # from grid to decimal observation
    def test_encode(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped

        # 3 x 3
        self.assertEqual(env.encode([[0] * 3] * 3), 0)
        self.assertEqual(env.encode([[2] * 3] * 3), 19682)
        self.assertEqual(env.encode([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), 6561)
        self.assertEqual(env.encode([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 1)
        self.assertEqual(env.encode([[0, 2, 1], [2, 1, 1], [1, 2, 2]]), 18618)

        # 4 x 4
        self.assertEqual(env.encode([[0] * 4] * 4), 0)
        self.assertEqual(env.encode(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), 1)

    # from decimal observation to grid
    def test_decode(self):
        # 3 x 3
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        self.assertEqual(env.unwrapped.decode(0), [[0] * 3] * 3)
        self.assertEqual(env.unwrapped.decode(19682), [[2] * 3] * 3)
        self.assertEqual(env.unwrapped.decode(6561), [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(env.unwrapped.decode(1), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env.unwrapped.decode(18618), [[0, 2, 1], [2, 1, 1], [1, 2, 2]])

        # 4 x 4
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1', size=4).unwrapped
        self.assertEqual(env.decode(0), [[0] * 4] * 4)
        self.assertEqual(env.decode(1), [[1, 0, 0, 0], [
            0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_reset(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        env.reset()
        self.assertEqual(env.s, 0)

    def test_preset(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        state = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
        env.s = env.encode(state)
        self.assertEqual(env.s, 6561)

    def test_turn(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped  # 3x3
        env.reset()

        self.assertEqual(env.turn([0, 0]), True)
        self.assertEqual(env.decode(env.unwrapped.s), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env.turn([0, 0]), False)
        self.assertEqual(env.turn([1, 0]), False)

        self.assertEqual(env.turn([1, 1]), True)
        self.assertEqual(env.decode(env.unwrapped.s), [[1, 2, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env.turn([1, 1]), False)
        self.assertEqual(env.turn([0, 1]), False)

    def test_render(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        env.reset()
        env.turn([0, 0])
        env.turn([1, 1])

        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            env.render()
            self.assertEqual(fakeOutput.getvalue().strip(),
                             '|O|X| |\n| | | |\n| | | |')

    def testis_win(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        env.reset()
        self.assertEqual(env.is_win(1), False)
        self.assertEqual(env.is_win(0), True)

        env.turn([0, 0])
        env.turn([0, 1])
        env.turn([0, 2])
        self.assertEqual(env.is_win(2), False)
        self.assertEqual(env.is_win(1), True)

        env.reset()
        env.turn([1, 0])
        env.turn([1, 1])
        env.turn([1, 2])
        self.assertEqual(env.is_win(1), False)
        self.assertEqual(env.is_win(2), True)

        env.reset()
        env.turn([0, 0])
        env.turn([0, 3])
        env.turn([0, 6])
        self.assertEqual(env.is_win(1), True)

        env.reset()
        env.turn([0, 0])
        env.turn([0, 4])
        env.turn([0, 8])
        self.assertEqual(env.is_win(1), True)

        env.s = 8260
        self.assertEqual(env.is_win(2), True)

        env.s = env.encode([[1, 0, 0], [0, 1, 1], [0, 1, 0]])
        self.assertEqual(env.is_win(1), False)

        env.s = env.encode([[1, 2, 1], [2, 1, 1], [1, 2, 2]])
        self.assertEqual(env.is_win(1), True)

    def test_is_full(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        env.reset()

        env.s = env.encode([[0] * 3] * 3)
        self.assertEqual(env.is_full(), False)

        env.s = env.encode([[1] * 3] * 3)
        self.assertEqual(env.is_full(), True)

        env.s = env.encode([[2] * 3] * 3)
        self.assertEqual(env.is_full(), True)

    def test_step(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        env.reset()

        # normal move
        (observation, reward, terminated, truncated, info) = env.step([0, 0])
        self.assertEqual(env.decode(observation), [[1, 0, 0], [0] * 3, [0] * 3])
        self.assertEqual(reward, env.reward_normal)
        self.assertEqual(terminated, False)
        self.assertEqual(truncated, False)
        self.assertEqual(info['info'], 'normal move')
        self.assertEqual(info['player'], 1)

        # violation move
        (observation, reward, terminated, truncated, info) = env.step([0, 0])
        self.assertEqual(env.decode(observation), [[1, 0, 0], [0] * 3, [0] * 3])
        self.assertEqual(reward, env.reward_violation)
        self.assertEqual(terminated, False)
        self.assertEqual(truncated, False)
        self.assertEqual(info['info'], 'invalid move')
        self.assertEqual(info['player'], 1)

        # winning move
        env.step([0, 1])
        (observation, reward, terminated, truncated, info) = env.step([0, 2])
        self.assertEqual(env.decode(observation), [[1, 1, 1], [0] * 3, [0] * 3])
        self.assertEqual(reward, env.reward_win)
        self.assertEqual(terminated, True)
        self.assertEqual(truncated, False)
        self.assertEqual(info['info'], 'winning move')
        self.assertEqual(info['player'], 1)

        # drawn move
        env.s = env.encode([[0, 2, 1], [2, 1, 1], [2, 1, 2]])
        (observation, reward, terminated, truncated, info) = env.step([0, 0])
        self.assertEqual(env.decode(observation), [
            [1, 2, 1], [2, 1, 1], [2, 1, 2]])
        self.assertEqual(reward, env.reward_drawn)
        self.assertEqual(terminated, True)
        self.assertEqual(truncated, False)
        self.assertEqual(info['info'], 'drawn move')
        self.assertEqual(info['player'], 1)

    def test_get_valid_moves(self):
        env: TictactoeEnv = gym.make('gym_tictactoe:tictactoe-v1').unwrapped
        env.reset()
        env.step([0, 0])
        env.step([0, 8])
        env.step([1, 1])

        self.assertEqual(env.get_valid_moves(), [2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    unittest.main()
