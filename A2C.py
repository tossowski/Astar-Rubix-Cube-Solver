from torch import nn
from torch.nn import functional as F
import torch
from collections import deque
import numpy as np
import random

class A2CAgent(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super().__init__()


        self.num_actions = num_actions
        self.critic_conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.critic_linear1 = nn.Linear(1944, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.actor_linear1 = nn.Linear(1944, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, inputs):

        value = F.relu(self.critic_conv1(inputs))
        value = torch.flatten(value)
        value = F.relu(self.critic_linear1(value))
        value = self.critic_linear2(value)
        
        policy = F.relu(self.actor_conv1(inputs))
        policy = torch.flatten(policy)
        policy_dist = F.relu(self.actor_linear1(policy))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=0)

        return value, policy_dist



    def computeDiscountedrewards(self, rewards, gamma=0.99):
        prev = 0

        for i in range(1, len(rewards) + 1):
            rewards[-i] += prev * gamma
            prev = rewards[-i]

        # rewards = torch.tensor(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std())
        
        return rewards
    
    def loss(self, neglogprobs, rewards, values, gamma = 0.99):
        rewards = self.computeDiscountedrewards(rewards)
        actorloss = 0
        criticloss = 0
        examples = 0
        for logprob, value, reward in zip(neglogprobs, values, rewards):
            advantage = reward  - value
            action_loss = logprob * advantage
            value_loss = (reward - value) ** 2
            actorloss += action_loss
            criticloss += value_loss
            examples += 1   
        return actorloss / examples + criticloss / examples

