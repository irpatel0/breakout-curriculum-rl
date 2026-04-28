import gymnasium as gym
import torch
import ale_py
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing
from agent import DQNAgent
from tqdm import tqdm

def render_env(env, policy, max_steps):
    obs, _ = env.reset()
    for i in tqdm(range(max_steps)):
        #Get the next action based on the current policy
        action = policy.take_action(obs)

        #Perform that action
        next_obs, reward, done, truncated, _ = env.step(action)

        #Update the policy
        policy.step(obs, action, reward, next_obs, done)


        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    env.close()


gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Breakout environment
env = gym.make("ALE/Breakout-v5", frameskip = 1, difficulty=0)
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)
env = FrameStackObservation(env, stack_size=4)

policy = DQNAgent(env.action_space)
render_env(env, policy, 200000)
