from gymnasium.envs.registration import register

register(
    id='tictactoe-v1',
    entry_point='gym_tictactoe.envs:TictactoeEnv',
)