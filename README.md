# gym-tic-tac-toe

Open AI Gym environment for Tic-Tac-Toe.

## Installation

Create virtual environment:

```
python -m venv venv
venv/Scripts/activate
```

Python 3.12+ compatibility:

```
pip install setuptools
```

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Make sure to have a look at the demo `demo_learning.py` where Q-Learning is used to train an agent.
Or take a look at `demo_random.py` where two random opponents play against each other.

### Initialization

```python
import gymnasium as gym

env = gym.make('gym_tictactoe:tictactoe-v1')
env.reset()

env.render()
# | | | |
# | | | |
# | | | |
```

### Make a move

The board is indexed as follows:

```python
# |0|1|2|
# |3|4|5|
# |6|7|8|
```

To make a move, call the `step` - function:

```python
(observation, reward, done, info) = env.step([0, 3])  # 0 for player 1 and position 3
# (27, 0, False, 'normal move')

env.unwrapped.render()
# | | | |
# |O| | |
# | | | |

env.step([1, 2])  # 1 for player 2 and position 2
env.unwrapped.render()
# | | |X|
# |O| | |
# | | | |
```

### State representation

```python
preset = [[0, 1, 2], [0, 0, 0], [1, 0, 2]]
env.unwrapped.s = env.unwrapped.encode(preset)
env.unwrapped.render()
# | |O|X|
# | | | |
# |O| |X|

print(env.unwrapped.s)
# 13872

board = env.unwrapped.decode(env.unwrapped.s)
# [[0, 1, 2], [0, 0, 0], [1, 0, 2]]
```

###

## Quick demo

From `demo_random.py`:

```python
import gymnasium as gym
import itertools


def main():
    env = gym.make('gym_tictactoe:tictactoe-v1')

    observation, info = env.reset()
    while True:
        player = next_player()
        position = env.action_space.sample()[1]
        observation, reward, done, info = env.step((player, position))
        env.render()

        if done:
            print(info)
            break


next_player = itertools.cycle([0, 1]).__next__
if __name__ == '__main__':
    main()

```

## Contributing

Please make sure to update tests (`tests.py`) as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
