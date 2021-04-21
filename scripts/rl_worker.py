#!/usr/bin/env python3
import sys
import os
import gym

import matplotlib.pyplot as plt
import datetime
import rospy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

import environment
from custom_policy import CustomEpsGreedy

ENV_NAME = 'RobotEnv-v0'

file_path = __file__
dir_path = file_path[:(len(file_path) - len('scripts/rl_worker.py'))]
MODELS_PATH = dir_path + 'models/'   # model save directory
FIGURES_PATH = dir_path + 'figures/'


if __name__ == '__main__':
    rospy.init_node('rl_worker', anonymous=True)
    mode = rospy.get_param('mode', 'train')
    weights_filename = rospy.get_param('weights_filename', 'dpg_RobotEnv-v0_weights.h5f')

    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(nb_actions))
    print(model.summary())

    memory = SequentialMemory(limit=1200, window_length=1)
    policy = CustomEpsGreedy(max_eps=0.6, min_eps=0.1, eps_decay=0.9997)

    agent = DQNAgent(
        nb_actions=nb_actions,
        model=model,
        memory=memory,
        policy=policy,
        gamma=0.99,
        batch_size=64)

    agent.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])
   
    if mode == 'train':

        #tensorboard_callback = TensorBoard(log_dir="~/tflog/")
        # early_stopping = EarlyStopping(monitor='episode_reward', patience=0, verbose=1)
        history = agent.fit(env,
                            nb_steps=1200,
                            visualize=False,
                            nb_max_episode_steps=300,
                            log_interval=300,
                            verbose=1)

        env.close()

        dt_now = datetime.datetime.now()
        agent.save_weights(
            MODELS_PATH + 'dpg_{}_weights_{}{}{}.h5f'.format(ENV_NAME, dt_now.month, dt_now.day, dt_now.hour),
            overwrite=True)

        fig = plt.figure()
        plt.plot(history.history['episode_reward'])
        plt.xlabel("episode")
        plt.ylabel("reward")

        plt.savefig(FIGURES_PATH + 'learning_results_{}{}{}.png'
                    .format(dt_now.month, dt_now.day, dt_now.hour))

    elif mode == 'test':
        if weights_filename:
            agent.load_weights(MODELS_PATH + weights_filename)
            agent.test(env, nb_episodes=5, visualize=False)
