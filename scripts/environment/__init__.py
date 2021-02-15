from gym.envs.registration import register

register(
    id='RobotEnv-v0',
    #directory:class name
    entry_point='environment.robot_rl_env:RobotEnv'
)
