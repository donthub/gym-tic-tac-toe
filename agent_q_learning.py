import pickle
import random

import numpy as np

AGENT_Q_LEARNING_Q_TABLE_FILENAME = "agent_q_learning_q_table.p"


class AgentQLearning:

    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.8, from_scratch=False):
        self.env = env
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.from_scratch = from_scratch
        self.Q = self.create_Q()

    def create_Q(self):
        """
        Initializes Q-Table, where:
            rows = states
            columns = actions
            entries = values = sum of accumulated expected reward

        Returns:
            zero-matrix m x n where:
                m = observation space
                n = action space
        """

        if self.from_scratch:
            return np.zeros([self.env.observation_space.n, int(self.env.action_space.nvec[1])])
        else:
            try:
                print('Loading Q-Table')
                return pickle.load(open(AGENT_Q_LEARNING_Q_TABLE_FILENAME, "rb"))
            except IOError:
                print('Could not find file. Starting from scratch')
                return np.zeros([self.env.observation_space.n, int(self.env.action_space.nvec[1])])

    def play_one(self, opponent, render=False, update=True, first=True, explore=True):
        """
        Agent plays one match against an opponent.

        Args:
            opponent: function (env) -> action, returns action given an environment
            render=False: Whether to display the field after each move
            update=True: Whether to update the Q-table
            first=True: If agent starts the game
            explore=True: If exploration is allowed for agent decision making

        Returns:
            Tuple of (a, b) where
                a = (updated) Q-Table
                b = outcome of the move = {None, 'win', 'loss', 'draw'}
        """

        state, info = self.env.reset()

        done = False
        agent_moved = False
        opponent_moved = False
        agent_reward = 0
        old_value = None

        # Play
        while not done:
            # Agent moves, skip in first round if second
            if first or opponent_moved:
                action = self.agent_move(state, explore)
                next_state, agent_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                agent_moved = True
                old_value = self.Q[state, action]

                [print('Agent moved:'), self.env.render(), print()] if render else None

            # Opponent makes a move, but only if not done
            if not done:
                opponent_action = opponent(self.env)
                next_state, opponent_reward, terminated, truncated, info = self.env.step(opponent_action)
                done = terminated or truncated
                opponent_moved = True

                [self.env.render(), print()] if render else None
            else:
                opponent_reward = 0

            # update Q Table but only after opponent has moved
            if update and agent_moved:
                agent_reward -= opponent_reward
                next_value = np.max(self.Q[next_state])
                temp_diff = agent_reward + self.gamma * next_value - old_value
                self.Q[state, action] = old_value + self.alpha * temp_diff
            state = next_state

        # Game finished, get outcome for agent
        outcome = None
        if self.env.unwrapped.is_win(1):
            outcome = 'win'
        elif self.env.unwrapped.is_win(2):
            outcome = 'loss'
        else:
            outcome = 'draw'

        return self.Q, outcome

    def agent_move(self, state, explore, player=0):
        """
        Agent move decision.
        Given the state and the values of the Q table, agent chooses action with maximum value.
        Chance to also ignore Q-table and to explore new actions.

        Args:
            state: current observed state of the environment
            explore: True/False to tell if able to explore

        Returns:
            action to take in form (a, b) where:
                a = player
                b = field to place stone by index
        """

        if explore and random.uniform(0, 1) < self.epsilon:
            return player, self.env.action_space.sample()[1]  # explore action space
        else:
            return player, np.argmax(self.Q[state])  # exploit learned values

    def export(self):
        pickle.dump(self.Q, open(AGENT_Q_LEARNING_Q_TABLE_FILENAME, "wb"))
