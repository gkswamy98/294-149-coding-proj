import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from gym import spaces
from comedy_of_the_commons import COTCEnv

# Copied from MADDPG
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for Comedy of the Commons")
    # Env specific params
    parser.add_argument("--num-agents", type=int, default=10, help="number of agents")
    parser.add_argument("--num-ddpg", type=int, default=1, help="number of ddpg agents")
    parser.add_argument("--num-maddpg", type=int, default=9, help="number of maddpg agents")
    parser.add_argument("--save-suffix", type=str, default="", help="suffix to make sure saving is distinct")
    # Setup params
    parser.add_argument("--max-episode-len", type=int, default=5, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    return parser.parse_args()

# Copied from MADDPG
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

# Modified from MADDPG
def get_trainers(num_agents, num_ddpg, num_maddpg):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    act_space_n = [spaces.Discrete(num_agents) for _ in range(num_agents)]
    obs_shape_n = [(num_agents, ) for _ in range(num_agents)]
    for i in range(num_agents):
        if i < num_ddpg:
            trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, act_space_n, i, arglist,
            local_q_func=(True)))
        else:
            trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, act_space_n, i, arglist,
            local_q_func=(False)))
    return trainers

# Modified from MADDPG
def train(arglist):
    with U.single_threaded_session():
        num_agents = arglist.num_agents
        num_ddpg = arglist.num_ddpg
        num_maddpg = arglist.num_maddpg

        print('Using {0} DDPG Agents and {1} MADDPG Agents.'.format(num_ddpg, num_maddpg))
        assert num_agents == num_ddpg + num_maddpg, "Numbers of ddpg and maddpg agents do not add up to total number of agents"

        trainers = get_trainers(num_agents, num_ddpg, num_maddpg)
        env = COTCEnv(num_agents, arglist.max_episode_len)

        U.initialize()

        obs_n = [np.zeros(num_agents) for _ in range(num_agents)]
        episode_step = 0
        train_step = 0
        agent_rewards = [[0.0] for _ in range(num_agents)]  # individual agent reward
        ddpg_rewards = []
        maddpg_rewards = []
        ddpg_points = []
        maddpg_points = []
        t_start = time.time()

        print('Starting iterations...')

        while len(ddpg_rewards) <= arglist.num_episodes:
            print("--------------------STEP {0}--------------------".format(train_step))
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)] # Gives us dist over n choices

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # record data on points
            points = new_obs_n[0]
            ddpg_avg_points = np.mean([points[i] for i in range(num_ddpg)])
            ddpg_points.append(ddpg_avg_points)
            maddpg_avg_points = np.mean([points[i] for i in range(num_ddpg, num_agents)])
            maddpg_points.append(maddpg_avg_points)
            print("DDPG AVG POINTS:", ddpg_avg_points)
            print("MADDPG AVG POINTS:", maddpg_avg_points)

            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                ddpg_avg_reward = np.mean([agent_rewards[i][-1] for i in range(num_ddpg)])
                ddpg_rewards.append(ddpg_avg_reward)
                maddpg_avg_reward = np.mean([agent_rewards[i][-1] for i in range(num_ddpg, num_agents)])
                maddpg_rewards.append(maddpg_avg_reward)
                print("DDPG AVG REWARD:", ddpg_avg_reward)
                print("MADDPG AVG REWARD:", maddpg_avg_reward)

                for a in agent_rewards:
                    a.append(0)

            # increment global step counter
            train_step += 1

            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

        # save data
        np.save('./data/agent_rewards_{0}.npy'.format(arglist.save_suffix), agent_rewards)
        np.save('./data/ddpg_rewards_{0}.npy'.format(arglist.save_suffix), ddpg_rewards)
        np.save('./data/maddpg_rewards_{0}.npy'.format(arglist.save_suffix), maddpg_rewards)
        np.save('./data/ddpg_points_{0}.npy'.format(arglist.save_suffix), ddpg_points)
        np.save('./data/maddpg_points_{0}.npy'.format(arglist.save_suffix), maddpg_points)



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)