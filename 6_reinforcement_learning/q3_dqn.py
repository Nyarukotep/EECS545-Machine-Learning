import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

random.seed(403)
np.random.seed(403)
torch.manual_seed(403)


env = gym.make('CartPole-v1')
env.seed(403)
env = env.unwrapped

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
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
        self.fc1 = nn.Linear(env.observation_space.shape[0], 50)  
        self.fc1.weight.data.normal_(0, 0.1) 
        self.fc2 = nn.Linear(50, env.action_space.n,)
        self.fc2.weight.data.normal_(0, 0.1)



    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out


def select_action(state, policy_net, eps_end, eps_start, eps_decay, steps_done, device):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    
    ########### TODO: Epsilon-greedy action selection ##############
    ### with probability eps_threshold, take random action       ###
    ### with probability 1-eps_threshold, take the greedy action ###
    ################################################################
    if sample > eps_threshold:
        actval = policy_net(state)
        action = torch.max(actval,1)[1].view(1,1)
        
    else:
        action = torch.tensor([[np.random.randint(0, env.action_space.n)]],device=device)
       
    return action


def optimize(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    """Learning step of DQN."""

    if len(memory) < batch_size:
        return

    # Converts batch-array of transitions to Transition of batch-arrays.
    # (see https://stackoverflow.com/a/19343/3343043 for detailed explanation)
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    ########### TODO: Compute the current Q(s_t, a_t) #########
    ### Q-value computed using the policy network           ###
    ### (i.e., the current q-network)                       ###
    ### then we select the columns of actions taken.        ###
    ### These are the actions which would've been taken     ###
    ### for each batch state according to policy_net        ###
    ###########################################################
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    ########### TODO: Compute the target Q(s_t, a_t) for non-final state s_t ######
    ### Q-value computed using the target network                               ###
    ### (i.e., the older q-network) using Bellman equation.                     ###
    ### Hint: Select the best q-value using max(1)[0].                          ###
    ### Hint2: Use "non_final_mask" and "non_final_next_states" so that         ###
    ### resulting target value is either target Q value or 0 (final state)      ###
    ###############################################################################
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    target_state_action_values = reward_batch + gamma*next_state_values
  

    ########### TODO: Compute loss using either l2 or smooth l1 #######
    ### Note: you can use pytorch loss functions (e.g. F)
    ###################################################################
    loss = F.l1_loss(state_action_values, target_state_action_values)

    # Update parameters
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def plot_durations(episode_durations, save_path):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Plot 100-episode running averages
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(save_path)
    plt.show()
    plt.close()



def main():
    # env = gym.make('CartPole-v1')
    # env.seed(403) # Do not change the seed

    # Env info
    print("Action_space", env.action_space)
    print("Observation_space", env.observation_space)

    # Use GPU when available (optional)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameters
    BATCH_SIZE = 256
    GAMMA = 0.99
    EPS_START = 0.3
    EPS_END = 0.01
    EPS_DECAY = 10000
    TARGET_UPDATE = 20 # The target network update frequency

    num_episodes = 1000

    # Build the network and the optimizer
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.002)
    memory = ReplayMemory(10000)
    steps_done = 0
    episode_durations = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32)
        env.render(mode='rgb_array')
        for t in count():
            # Select and perform an action
            action = select_action(state, policy_net, eps_end=EPS_END,
                                   eps_start=EPS_START, eps_decay=EPS_DECAY,
                                   steps_done=steps_done, device=device)

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
            reward = torch.tensor([reward], device=device)


            if done:
                next_state=None

            steps_done += 1
            # env.render(mode='rgb_array')

            ########### TODO: Store the transition in memory ###########
            memory.push(state, action, next_state, reward)
            # ----------------------------------------------------------

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize(policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                     memory=memory, batch_size=BATCH_SIZE, gamma=GAMMA, device=device)
            if done:
                episode_durations.append(t + 1)
                print('episode', i_episode, 'duration', episode_durations[-1])
                break

        # Update the target network, copying all weights and biases from the policy network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.close()
    plot_durations(episode_durations, 'dqn_reward.png')


if __name__ == '__main__':
    main()