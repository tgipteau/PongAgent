import torch
from collections import namedtuple
import random
import yaml

# loading config.yaml as default configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

    
class Network(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x2 = self.fc2(x)
        return x2  # logits -> argmax
    
class Agent:
    def __init__(self, state_size, action_size, cfg=config):
        self.state_size = state_size
        self.action_size = action_size

        self.training_reward = 0
        self.model = Network(state_size, action_size, cfg["hidden_size"]).to(device)
        self.model.eval()


    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = self.model(state)
        
        action = torch.argmax(scores).item()
        return action

        

