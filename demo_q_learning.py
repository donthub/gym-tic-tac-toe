import random

import gymnasium as gym
import numpy as np

from agent_q_learning import AgentQLearning


def get_next_envs(env, turn):
    """
    Get list of possible next environments given a player at turn.

    Args:
        env: The environment to create the Q-table for
        turn: {0, 1}, which player is at turn

    Returns:
        list of possible gym_tictactoe.envs.tictactoe_env.TictactoeEnv
            by looking at all possible moves given the player at turn
    """

    free_moves = env.unwrapped.get_valid_moves()
    next_envs = []
    for free_move in free_moves:
        env_copy = gym.make('gym_tictactoe:tictactoe-v1', size=size, num_winning=num_winning)
        env_copy.unwrapped.s = env.unwrapped.s
        env_copy.step((turn, free_move))
        next_envs.append(env_copy)
    return next_envs, free_moves


def minmax_state_value(env, player, turn):
    """
    Calculates the value of a board state given a player's view and a player at turn.
    Makes use of minmax algorithm.

    Args:
        env: The environment
        player: {0, 1}, the player's view
        turn: {0, 1}, which player is at turn

    Returns:
        the value of the current state of the environment in the eyes of a player.
            int {-1, 0, 1} where:
                1 is best value
                -1 is worst value
    """

    # end cases
    if env.unwrapped.is_win(player + 1):
        return 1
    elif env.unwrapped.is_win(int(not player) + 1):
        return -1
    elif env.unwrapped.is_full():
        return 0

    # build possible next boards given which player is at turn
    (next_envs, _) = get_next_envs(env, turn)

    combine_func = max if turn == player else min
    return combine_func([minmax_state_value(next_env, player, int(not turn)) for next_env in next_envs])


def opponent_minmax(env, player=1):
    """
    Opponent for the agent that chooses the move that will result in the next state with the highest value
    given by a minmax algorithm.

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player = 1
            b = field to place stone by index
    """
    (next_envs, free_moves) = get_next_envs(env, player)
    return (
        player,
        free_moves[np.argmax([minmax_state_value(next_env, player, int(not player)) for next_env in next_envs])])


def opponent_random(env, player=1):
    """
    Opponent for the agent that chooses a random position on the board.
    Might result in invalid moves which will have no effect

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    return player, env.action_space.sample()[1]


def opponent_random_better(env, player=1):
    """
    Opponent for the agent that chooses a random free position on the board.
    Therefore only chooses a valid action

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    valid_moves = env.unwrapped.get_valid_moves()
    return player, random.choice(valid_moves)


def opponent_human(env, player=1):
    """
    Human opponent. Asks for input via terminal until a valid action is transmitted

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    action = [player, None]
    while action == [player, None] or not env.action_space.contains(action):
        print('Pick a move: ', end='')
        user_input = input()
        action[1] = int(user_input) - 1 if user_input.isdigit() else None
    return tuple(action)


def train(agent, epochs, opponents):
    """
    Train the agent

    Args:
        agent: Agent
        epochs: Number of epochs to train
        opponents: list of opponents as functions (env) -> action to play against

    Returns:
        updated Q-table
    """

    is_first = True
    for i in range(epochs):
        agent.play_one(random.choice(opponents), first=is_first)
        is_first = not is_first

    return agent.Q


def test(agent, epochs, opponent, render=False):
    """
    Evaluate the performance of the agent by playing against one opponent.
    No exploration, no updates to Q-table.

    Args:
        agent: Agent
        epochs: Number of epochs to train
        opponent: opponent as function (env) -> action to play against
        render=False: If board shall be rendered and game outcome printed

    Returns:
        Tuple (wins, losses, draws)
    """

    outcome_list = [None for i in range(epochs)]
    is_first = True
    for i in range(epochs):
        (_, outcome_agent) = agent.play_one(opponent, render=render, first=is_first, update=False, explore=False)

        if render:
            if outcome_agent == 'win':
                print('You lost')
            elif outcome_agent == 'loss':
                print('You won')
            else:
                print('Drawn')

        outcome_list[i] = outcome_agent
        is_first = not is_first

    wins = sum(1 for i in outcome_list if i == 'win')
    losses = sum(1 for i in outcome_list if i == 'loss')
    drawns = sum(1 for i in outcome_list if i == 'drawn')

    return wins, losses, drawns


def main():
    """
    Main function:
        1. Create environment
        2. Preload Q-table if present, otherwise create from scratch
        3. Train for n epochs playing against random opponent
        4. Test for n/10 epochs and show performance
        5. Play against human

    Args:
        -

    Returns:
        -
    """

    # create environment
    env = gym.make('gym_tictactoe:tictactoe-v1', size=size, num_winning=num_winning)

    agent = AgentQLearning(env, epsilon=epsilon, alpha=alpha, gamma=gamma)

    # Play against a random player and learn
    print(f'Learning for {epochs} epochs')
    train(agent, epochs, [opponent_random, opponent_random_better])
    print(f'Finished!')
    print('Saving Q-Table')
    agent.export()

    # Test
    print(f'Testing for {int(epochs / 10)} epochs')
    wins, losses, draws = test(agent, int(epochs / 10), opponent_random_better)
    print(f'Wins: {wins}, Losses: {losses}, Drawns: {draws}')

    # Play against human player
    print(f'Playing 4 games against human player')
    losses, wins, draws = test(agent, 4, opponent_human, render=True)
    print(f'Wins: {wins}, Losses: {losses}, Drawns: {draws}')


# Hyperparameters
epsilon = 0.1  # exploration rate
alpha = 0.1  # learning rate
gamma = 0.8  # discount factor
epochs = 500000  # number of games played while training

# other
from_scratch = False

# Board settings
size = 3
num_winning = 3

if __name__ == '__main__':
    main()
