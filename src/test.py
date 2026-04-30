import argparse
import ale_py
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm
import yaml
from agent import DQNAgent

#register env
gym.register_envs(ale_py)

def create_env(env_config, difficulty, render_game=False):
    env = gym.make("ALE/Breakout-v5", difficulty=difficulty, frameskip=1, render_mode = "human" if render_game else None)
    env = AtariPreprocessing(env, screen_size=env_config["screen_size"], grayscale_obs=env_config["grayscale_obs"])
    env = FrameStackObservation(env, stack_size=env_config["stack_size"])
    return env

def test_DQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of episodes to test the agent")
    parser.add_argument("--difficulty", type=int, choices=[0, 1], default=0, help="Difficulty level of the environment (0 for easy, 1 for hard)")
    parser.add_argument("--render", action="store_true", help="Render the game during testing")
    
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    #Create environment and agent
    env = create_env(config["env"], difficulty=args.difficulty, render_game=args.render)
    agent = DQNAgent(env.action_space, config["agent"], config["training"]["steps"])

    #Load the parameters into DQN
    agent.policy_net.load_state_dict(torch.load(args.model_path, map_location=agent.device))
    agent.policy_net.eval()
    #Use a low epsilon, we want the almost always use the model for actions
    #small chance of random action so we don't get stuck
    agent.epsilon = 0.01

    #track all episode rewards
    total_rewards = []

    #run [test_episodes] games
    for episode in tqdm(range(args.test_episodes)):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action = agent.take_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            #add each step reward to the cumulative episode reward
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()

    #Print mean and standard deviation of rewards
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Average Reward over {args.test_episodes} episodes: {avg_reward}")
    print(f"STD. of Reward over {args.test_episodes} episodes: {std_reward}")

if __name__ == "__main__":
    test_DQN()