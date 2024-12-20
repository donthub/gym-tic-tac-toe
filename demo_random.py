import itertools

import gymnasium as gym


def main():
    env = gym.make('gym_tictactoe:tictactoe-v1')

    observation, info = env.reset()
    while True:
        player = next_player()
        position = env.action_space.sample()[1]
        observation, reward, terminated, truncated, info = env.step((player, position))
        done = terminated or truncated
        env.render()

        if done:
            print(info)
            break


next_player = itertools.cycle([0, 1]).__next__

if __name__ == '__main__':
    main()
