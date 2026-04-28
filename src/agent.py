import torch
import torch.optim as optim
import random
import math
from model import AtariDQN
from replayBuffer import ReplayBuffer
import torch.nn.functional as F
import numpy as np

class DQNAgent:
    def __init__(self, action_space, config, total_steps):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space

        #hyperparams
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]

        #Epsilon represents how likely we are to take a random action instead of the action with the highest Q-val
        #Encourage random exploring to start, slowly decay until 10%
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon = self.epsilon_start
        self.decay_steps = total_steps * config["decay_proportion"]
        self.target_update_freq = config["target_update_freq"]
        self.learning_rate = config["learning_rate"]
        self.buffer_capacity = config["buffer_capacity"]
        self.train_buffer = config["train_buffer"]

        self.policy_net = AtariDQN(stacked_frames=4, num_actions=action_space.n).to(self.device)
        self.target_net = AtariDQN(stacked_frames=4, num_actions=action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(capacity=self.buffer_capacity)

        self.num_steps = 0

    def load_model(self, path, start_step):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.num_steps = start_step
        
        curr_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.num_steps / self.decay_steps)
        self.epsilon = max(self.epsilon_end, curr_epsilon)


    def take_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
                action_values = self.policy_net(state)
                return torch.argmax(action_values, 1).item()
            
    def step(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

        self.num_steps += 1

        if len(self.memory) < self.train_buffer:
            return
        
        if self.num_steps % 4 == 0:
            self.optimize()
        
        self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.decay_steps
        self.epsilon = max(self.epsilon_end, self.epsilon)

        if self.num_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def optimize(self):

        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(self.device)

        policy_Q_guess = self.policy_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            target_Q_guess = self.target_net(batch_next_state).max(1)[0].detach().unsqueeze(1)
        expected_Q = batch_reward + (self.gamma * target_Q_guess * (1 - batch_done))

        loss = F.smooth_l1_loss(policy_Q_guess, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


