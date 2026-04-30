import random
from collections import deque

#Replay buffer for storing past transitions
class ReplayBuffer:

    #init
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    #add a transition to the buffer
    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    #sample [batch_size] transitions
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)