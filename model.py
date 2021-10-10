from pygame.image import load
import torch
from torch.functional import tensordot
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

class Liner_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.L1 = nn.Linear(input_size,hidden_size)
        self.L2 = nn.Linear(hidden_size, output_size)


    def forward(self,x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)

        return x
    
    def save(self, fname= 'model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path = os.path.join(model_path,fname)

        torch.save(self, model_path)

class Trainer:
    
    def __init__(self, model, lr, gamma ) -> None:
        
        self.lr = lr
        self.model = model
        self.gamma = gamma 

        self.optimizer = optim.Adam(model.parameters(), lr= self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, nxt_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        nxt_state = torch.tensor(nxt_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.long)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            nxt_state = torch.unsqueeze(nxt_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)

            game_over = (game_over, )


        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(nxt_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()

        loss = self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()




