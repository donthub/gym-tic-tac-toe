import random
import sys

import gymnasium as gym
import numpy as np
from tqdm import tqdm

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
        env_copy = gym.make('gym_tictactoe:tictactoe-v1')
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


def train(agent, n_episodes, opponents):
    """
    Train the agent

    Args:
        agent: Agent
        n_episodes: Number of episodes to train
        opponents: list of opponents as functions (env) -> action to play against

    Returns:
        updated Q-table
    """

    is_first = True
    for _ in tqdm(range(n_episodes), file=sys.stdout):
        agent.play_one(random.choice(opponents), first=is_first)
        is_first = not is_first

    return agent.Q


def test(agent, n_episodes, opponent, render=False):
    """
    Evaluate the performance of the agent by playing against one opponent.
    No exploration, no updates to Q-table.

    Args:
        agent: Agent
        n_episodes: Number of episodes to train
        opponent: opponent as function (env) -> action to play against
        render=False: If board shall be rendered and game outcome printed

    Returns:
        Tuple (wins, losses, draws)
    """

    outcome_list = [None for i in range(n_episodes)]
    is_first = True
    for episode in range(n_episodes) if render else tqdm(range(n_episodes), file=sys.stdout):
        (_, outcome_agent) = agent.play_one(opponent, render=render, first=is_first, update=False, explore=False)

        if render:
            if outcome_agent == 'win':
                print('You lost')
            elif outcome_agent == 'loss':
                print('You won')
            else:
                print('Draw')

        outcome_list[episode] = outcome_agent
        is_first = not is_first
    sys.stderr.flush()

    wins = sum(1 for i in outcome_list if i == 'win')
    losses = sum(1 for i in outcome_list if i == 'loss')
    draws = sum(1 for i in outcome_list if i == 'draw')

    return wins, losses, draws


def main():
    """
    Main function:
        1. Create environment
        2. Preload Q-table if present, otherwise create from scratch
        3. Train for n episodes playing against random opponent
        4. Test for n/10 episodes and show performance
        5. Play against human

    Args:
        -

    Returns:
        -
    """

    print('Select mode:')
    print('1: Train')
    print('2: Train from scratch')
    print('3: Test')
    from_scratch = False
    do_train = True
    selected_mode = input('Selected mode: ')
    if selected_mode == str(2):
        from_scratch = True
    elif selected_mode == str(3):
        do_train = False
    n_episodes = 500_000
    if do_train:
        n_episodes = int(input(f'Number of episodes (default: {n_episodes}): ').strip() or str(n_episodes))

    # create environment
    env = gym.make('gym_tictactoe:tictactoe-v1')
    agent = AgentQLearning(env, from_scratch=from_scratch)

    # Play against a random player and learn
    if do_train:
        print(f'Learning for {n_episodes} episodes')
        train(agent, n_episodes, [opponent_random, opponent_random_better])
        print(f'Finished!')
        print('Saving Q-Table')
        agent.export()

    # Test
    n_episodes_test = int(n_episodes / 10)
    print(f'Testing for {n_episodes_test} episodes')
    wins, losses, draws = test(agent, n_episodes_test, opponent_random_better)
    print(f'Wins: {wins}, Losses: {losses}, Draws: {draws}')

    # Play against human player
    print(f'Playing 4 games against human player')
    env.close()
    env = gym.make('gym_tictactoe:tictactoe-v1')
    agent.env = env
    losses, wins, draws = test(agent, 4, opponent_human, render=True)
    print(f'Wins: {wins}, Losses: {losses}, Draws: {draws}')


if __name__ == '__main__':
    main()
