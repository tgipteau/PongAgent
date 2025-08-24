import torch
from collections import namedtuple
import random
import yaml

# loading config.yaml as default configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
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
    
class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Agent:
    def __init__(self, state_size, action_size, cfg=config):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayMemory(cfg["memory_capacity"])
        self.model = DQN(state_size, action_size, cfg["hidden_size"]).to(device)
        self.target_model = DQN(state_size, action_size, cfg["hidden_size"]).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])

        self.batch_size = cfg["batch_size"]
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.epsilon_min = cfg["epsilon_min"]
        self.update_target_every = cfg["update_target_every"]
        self.step_count = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.FloatTensor([s for s in batch.next_state if s is not None]).to(device)

        current_q_values = self.model(state_batch).gather(1, action_batch).squeeze()
        next_q_values = torch.zeros(self.batch_size, device=device)
        next_q_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        
        expected_q_values = reward_batch + (self.gamma * next_q_values)

        loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())