import torch
import numpy as np
import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import DQNAgent

gym.register_envs(ale_py)

def create_env(env_config, difficulty):
    env = gym.make("ALE/Breakout-v5", difficulty=difficulty, frameskip=1)
    env = AtariPreprocessing(env, screen_size=env_config["screen_size"], grayscale_obs=env_config["grayscale_obs"])
    env = FrameStackObservation(env, stack_size=env_config["stack_size"])
    return env

def train_DQN(agent, num_steps, start_step, env, pth_name, window_size, success_thresh, save_halfway):

    writer = SummaryWriter(log_dir=f"logs/{pth_name}")

    episode_reward = 0
    past_episode_rewards = []
    successful = False
    halfway_step = start_step + num_steps // 2

    obs, _ = env.reset()
    
    for step in tqdm(range(start_step, start_step + num_steps)):
        #Get the next action based on the current policy
        action = agent.take_action(obs)
        #Perform that action
        next_obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        #Update the policy
        agent.step(obs, action, reward, next_obs, done)

        #set obs to the new observation
        if done:
            obs, _ = env.reset()
            past_episode_rewards.append(episode_reward)
            if len(past_episode_rewards) > 50:
                past_episode_rewards.pop(0)

            avg_reward = np.mean(past_episode_rewards)
            writer.add_scalar(f"Rewards/Train_Ep_Rewards", episode_reward, step)
            writer.add_scalar(f"Rewards/Moving_Avg_Rewards", avg_reward, step)

            if not successful and avg_reward >= success_thresh:
                print(f"\n[Success] {pth_name} reached success threshold at step {step}")
                writer.add_text('Success', f'Reached success threshold at step {step}', step)
                successful = True

            episode_reward = 0
        else:
            obs = next_obs

        if save_halfway and step == halfway_step - 1:
            torch.save(agent.policy_net.state_dict(), f"checkpoints/{pth_name}_half.pth")

    torch.save(agent.policy_net.state_dict(), f"checkpoints/{pth_name}_final.pth")
    env.close()