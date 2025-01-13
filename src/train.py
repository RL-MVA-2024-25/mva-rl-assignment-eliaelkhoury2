from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population
import random

# Environment Setup
env_df = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
env_rd = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
state_dim = env_df.observation_space.shape[0]
nb_actions = env_df.action_space.n

# DQN Model Definition
class DQN(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons=256):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        )

    def forward(self, x):
        return self.layers(x)

# Project Agent
class ProjectAgent:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Parameters
        self.gamma = 0.98
        self.batch_size = 256
        self.nb_actions = nb_actions
        self.memory_capacity = 50000
        self.epsilon_min = 0.1
        self.epsilon_update = 500
        self.epsilon_step = (1.0 - self.epsilon_min) / 15000
        self.update_step = 500
        self.gradient_app = 5
        self.learning_rate = 0.001

        # Replay Buffer
        self.buffer = []
        self.buffer_index = 0

        # Models
        self.model = DQN(state_dim, self.nb_actions).to(device)
        self.up_model = deepcopy(self.model).eval()

        # Training Setup
        self.crit = nn.HuberLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model = DQN(state_dim, self.nb_actions).to(self.device)
        path = os.getcwd() + "/model.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def append_to_buffer(self, states, actions, rewards, next_states, dones):
        if len(self.buffer) < self.memory_capacity:
            self.buffer.append(None)
        self.buffer[self.buffer_index] = (states, actions, rewards, next_states, dones)
        self.buffer_index = (self.buffer_index + 1) % self.memory_capacity

    def sample_from_buffer(self):
        batch = random.sample(self.buffer, self.batch_size)
        return [torch.Tensor(np.array(x)).to(self.device) for x in zip(*batch)]

    def gradient_step(self):
        if len(self.buffer) >= self.batch_size:
            samples = self.sample_from_buffer()
            states, actions, rewards, next_states, dones = samples

            with torch.no_grad():
                max_next_q_values = self.up_model(next_states).max(dim=1)[0]
                target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

            predicted_q_values = self.model(states).gather(1, actions.long().unsqueeze(1))
            loss = self.crit(predicted_q_values, target_q_values.unsqueeze(1))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def train(self):
        env = env_df
        epsilon = 1.0
        step = 0
        episode = 1
        current_score = 0
        state, _ = env.reset()
        episode_cum_reward = 0
        test_score = 0
        test_score_pop = 0

        while episode <= 150 :
            
            if step > self.epsilon_update:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # Select greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, trunc, _ = env.step(action)
            self.append_to_buffer(state, action, reward, next_state, done)
            episode_cum_reward += reward

            for _ in range(self.gradient_app):
                self.gradient_step()

            # Update model
            if step % self.update_step == 0:
                self.up_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:

                test_score = evaluate_HIV(agent=self, nb_episode=1)
                test_score_pop = evaluate_HIV_population(agent=self, nb_episode=1)
                
                print("Episode : {:3d} , cum_reward : {:e}, test_score : {:e}, test_score_pop : {:e}".format(episode, episode_cum_reward, test_score, test_score_pop))

                if test_score + test_score_pop > current_score:
                    current_score = test_score + test_score_pop
                    self.save("model.pt")

                episode_cum_reward = 0

                env = env_df if random.random()<0.75 else env_rd
                state, _ = env.reset()
                episode += 1
            else:
                state = next_state

'''if __name__ == "__main__":
    ProjectAgent().train()'''