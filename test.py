import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class QAgent:
    def __init__(self, input_dim, n_actions, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.model = QNetwork(input_dim, n_actions)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self, state, action, td_error):
        state_tensor = torch.FloatTensor(state)
        predicted_q_values = self.model(state_tensor)
        target_q_values = predicted_q_values.clone().detach()
        target_q_values[0][action] += td_error  # Apply TD error

        # Update model
        loss = self.criterion(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

class VDN:
    def __init__(self, n_agents, input_dim, n_actions):
        self.agents = [QAgent(input_dim, n_actions) for _ in range(n_agents)]

    def choose_actions(self, states):
        return [agent.choose_action(state) for agent, state in zip(self.agents, states)]

    def learn(self, states, actions, team_reward, next_states):
        # Get Q-values for the current state and action for all agents
        current_q_values = [agent.model(torch.FloatTensor(state))[action].item() for agent, state, action in zip(self.agents, states, actions)]
        
        # For the next state, get max Q-values of next states from all agents
        next_max_q_values = [torch.max(agent.model(torch.FloatTensor(next_state))).item() for agent, next_state in zip(self.agents, next_states)]
        
        # Compute the joint target Q-value using the team reward
        joint_target_q_value = team_reward + self.agents[0].gamma * sum(next_max_q_values)
        
        # Calculate TD errors for individual agents
        td_errors = [joint_target_q_value - q for q in current_q_values]
        
        # Update each agent's Q-network based on its TD error
        for i, (agent, state, action, next_state) in enumerate(zip(self.agents, states, actions, next_states)):
            agent.learn(state, action, td_errors[i])

    def decay_epsilon(self):
        for agent in self.agents:
            agent.decay_epsilon()

# Example usage
n_agents = 2
input_dim = 5
n_actions = 3

env = ...  # Some multi-agent environment

vdn = VDN(n_agents, input_dim, n_actions)

for episode in range(1000):
    states = env.reset()
    done = False
    
    while not done:
        actions = vdn.choose_actions(states)
        next_states, team_reward, done, _ = env.step(actions)  # Assuming the environment provides a team reward
        vdn.learn(states, actions, team_reward, next_states)
        states = next_states
        vdn.decay_epsilon()






