import torch
import numpy as np
import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import DQNAgent

#register env
gym.register_envs(ale_py)

#Create the atari env
def create_env(env_config, difficulty):
    env = gym.make("ALE/Breakout-v5", difficulty=difficulty, frameskip=1)
    env = AtariPreprocessing(env, screen_size=env_config["screen_size"], grayscale_obs=env_config["grayscale_obs"])
    env = FrameStackObservation(env, stack_size=env_config["stack_size"])
    return env


def eval_DQN(agent, env_config, difficulty, num_episodes=100):
    #create a temporary new environment for fresh start
    env = create_env(env_config, difficulty)

    #Use a low epsilon, we want the almost always use the model for actions
    #small chance of random action so we don't get stuck
    #store the old epsilon so we can re-set it after
    stored_epsilon = agent.epsilon
    agent.epsilon = 0.05

    #track total reward from each game
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action = agent.take_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    #restore the agent epsilon and close the temp env
    agent.epsilon = stored_epsilon
    env.close()

    #return the avg reward
    return np.mean(total_rewards)

def train_DQN(agent, num_steps, start_step, env, pth_name, window_size, success_thresh, save_halfway, env_config, difficulty, eval_interval, eval_episodes):

    #writer for logging
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
        if done or truncated:
            obs, _ = env.reset()

            #Keep track of the rewards from the episode
            past_episode_rewards.append(episode_reward)

            #Use a sliding window; pop old games
            if len(past_episode_rewards) > window_size:
                past_episode_rewards.pop(0)

            #Add episode stats to writer
            avg_reward = np.mean(past_episode_rewards)
            writer.add_scalar(f"Rewards/Train_Ep_Rewards", episode_reward, step)
            writer.add_scalar(f"Rewards/Moving_Avg_Rewards", avg_reward, step)

            #If the avg reward within the window is above the threshold for the first time, log it
            if not successful and avg_reward >= success_thresh:
                print(f"\n[Success] {pth_name} reached success threshold at step {step}")
                writer.add_text('Success', f'Reached success threshold at step {step}', step)
                successful = True

            #reset episode reward
            episode_reward = 0
        else:
            obs = next_obs

        #Perform an evaluation every [eval_interval] steps and log results
        if (step + 1) % eval_interval == 0:
            eval_reward = eval_DQN(agent, env_config, difficulty, num_episodes=eval_episodes)
            writer.add_scalar(f"Rewards/Eval_Rewards", eval_reward, step)

        #Save the model if we are at the halfway step
        if save_halfway and step == halfway_step - 1:
            torch.save(agent.policy_net.state_dict(), f"checkpoints/{pth_name}_half.pth")

    #save the model after all steps
    torch.save(agent.policy_net.state_dict(), f"checkpoints/{pth_name}_final.pth")
    env.close()