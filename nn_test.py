import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

env = gym.make('CartPole-v0')

#use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.affine1 = nn.Linear(4, 40)
        self.affine2 = nn.Linear(40, 40)
        self.action_head = nn.Linear(40, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores)


model = DQN()
if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)
memory = ReplayMemory(10000)

GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 2000


steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                               math.exp(-1. * steps_done / EPS_DECAY)
    print(eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


episode_durations = []
def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title("Trainning...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

last_sync = 0
BATCH_SIZE = 64
def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    next_state_values = Variable(torch.zeros(BATCH_SIZE,1).type(Tensor))

    temp = [s for s in batch.next_state if s is not None]
    if not temp:
        non_final_next_states = Variable(torch.cat(temp), volatile=True)
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    # Compute V(s_{t+1}) for all next states.
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 10000
for i_episode in range(num_episodes):

    state = env.reset()
    state = torch.from_numpy(state).type(Tensor).unsqueeze(0) # state: torch.FloatTensor of size 1xns
    for t in count():
        # env.render()
        action = select_action(state)  # action: torch.LongTensor of size 1x1
        next_state, reward, done, info = env.step(action[0, 0])
        reward = Tensor([[reward]])  # reward: torch.FloatTensor oq size 1x1
        if done:
            next_state = None
            reward = Tensor([[-20]])  # reward: torch.FloatTensor oq size 1x1
        else:
            next_state = torch.from_numpy(next_state).type(Tensor).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
plt.show()






