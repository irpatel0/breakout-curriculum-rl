import torch
import torch.nn as nn
import torch.nn.functional as F


#Following the DQN architecture used in Mnih's 2015 paper
class AtariDQN(nn.Module):
    def __init__(self, stacked_frames, num_actions):
        super(AtariDQN, self).__init__()

        self.conv1 = nn.Conv2d(stacked_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        #7x7 feature map w/ 64 channels
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):

        #normalize pixel values [0, 1]
        x = x / 255.0 

        #conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        #flatten
        x = x.flatten(start_dim=1)

        #fully connected layers
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out
    