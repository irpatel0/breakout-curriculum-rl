import yaml
import gc
import torch
import os
from train import create_env, train_DQN
from agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter
from colorama import init, Fore, Style

#colorama init
init()

#delete the environment and agent objects, garbage collect, and empty GPU cache
#this is necessary so that we can train multiple agents in one run
def clear_memory(agent, env):
    del agent
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_experiment(config_path):

    # --------------SETUP --------------
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    agent_config = config["agent"]
    env_config = config["env"]

    total_steps = training_config["steps"]
    success_thresh = training_config["success_thresh"]
    window_size = training_config["moving_avg_reward_window"]
    eval_interval = total_steps // training_config["num_checkpoints"]
    eval_episodes = training_config["eval_episodes"]

    os.makedirs("checkpoints", exist_ok=True)

    # --------------TRAINING EASY--------------
    print(Fore.GREEN + "Training Easy" + Style.RESET_ALL)
    #create env & agent
    easy_env = create_env(config["env"], 0)
    easy_agent = DQNAgent(easy_env.action_space, agent_config, total_steps)
    #train
    train_DQN(easy_agent, total_steps, 0, easy_env, "base_easy_short", window_size, success_thresh, save_halfway=True, 
              env_config=env_config, difficulty=0, eval_interval=eval_interval, eval_episodes=eval_episodes)
    #cleanup
    clear_memory(easy_agent, easy_env)

    # --------------TRAINING HARD--------------
    print(Fore.RED + "Training Hard" + Style.RESET_ALL)
    #create env & agent
    hard_env = create_env(config["env"], 1)
    hard_agent = DQNAgent(hard_env.action_space, agent_config, total_steps)
    #train
    train_DQN(hard_agent, total_steps, 0, hard_env, "base_hard_short", window_size, success_thresh, save_halfway=True, 
              env_config=env_config, difficulty=1, eval_interval=eval_interval, eval_episodes=eval_episodes)
    #cleanup
    clear_memory(hard_agent, hard_env)

    # --------------TRAINING CURRICULUM (EASY -> HARD) --------------
    print(Fore.GREEN + "Training Curriculum (Easy -> Hard)" + Style.RESET_ALL)
    #create env & agent, load the model weights
    curriculum_env = create_env(config["env"], 1)
    curriculum_agent = DQNAgent(curriculum_env.action_space, agent_config, total_steps)
    curriculum_agent.load_model("checkpoints/base_easy_short_half.pth", start_step=total_steps//2)
    #train
    train_DQN(curriculum_agent, total_steps//2, total_steps//2, curriculum_env, "curriculum_short", window_size, success_thresh, save_halfway=False, 
              env_config=env_config, difficulty=1, eval_interval=eval_interval, eval_episodes=eval_episodes)
    #cleanup
    clear_memory(curriculum_agent, curriculum_env)

    # --------------TRAINING REVERSE CURRICULUM (HARD -> EASY) --------------
    print(Fore.RED + "Training Reverse Curriculum (Hard -> Easy)" + Style.RESET_ALL)
    #create env & agent, load the model weights
    reverse_curriculum_env = create_env(config["env"], 0)
    reverse_curriculum_agent = DQNAgent(reverse_curriculum_env.action_space, agent_config, total_steps)
    reverse_curriculum_agent.load_model("checkpoints/base_hard_short_half.pth", start_step=total_steps//2)
    #train
    train_DQN(reverse_curriculum_agent, total_steps//2, total_steps//2, reverse_curriculum_env, "reverse_curriculum_short", window_size, success_thresh, save_halfway=False, 
              env_config=env_config, difficulty=0, eval_interval=eval_interval, eval_episodes=eval_episodes)
    #cleanup
    clear_memory(reverse_curriculum_agent, reverse_curriculum_env)

if __name__ == "__main__":
    run_experiment(config_path="config.yaml")