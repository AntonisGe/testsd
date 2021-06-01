import torch
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class network(nn.Module):
    def __init__(self):
                
        super(network,self).__init__()
              
        self.fc1 = nn.Linear(4,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,3)
        
    def forward(self,x):
    
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = torch.softmax(self.fc3(out),dim=-1)
        
        return out
        
class agent:
    def __init__(self):
        self.gamma = 0
        self.learning_rate = 0.004
        self.policy = network()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
    def choose_action(self,x):
      
        probs = self.policy.forward(x)
        distribution = Categorical(probs = probs)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action = action.cpu().detach().numpy()[0]       
        
        return action,log_prob
        
    def update(self,rewards,log_probs):
    
        # p_loss = 0.0
        # G= 0
        # t = len(rewards)-1
        # while t != -1:
            # log_prob = log_probs[0]
            # G = rewards[t] + self.gamma*G 
            # G = torch.tensor(G,dtype = torch.float)
            # p_loss -= G*log_prob
            # t -= 1

        # for simplicity I have just one reward and log prob per episode
        
        reward = rewards[0]
        log_prob = log_probs[0]
        
        G = torch.tensor(reward,dtype=torch.float)
        
        p_loss = -G*log_prob

        p_loss.backward()
        self.policy_optim.step()
        self.policy_optim.zero_grad()
        
        
ag = agent()    
for itter in range(500):
    a = int(np.random.choice(2))
    b = int(np.random.choice(2))
    c = int(np.random.choice(2))
    d = int(np.random.choice(2))

    x = torch.tensor([[a,b,c,d]],dtype=T.float32)
    
    action,log_prob = ag.choose_action(x)

    if a == 1:
        if action == 0:
            reward = 1
        else:   
            reward = 0
    else:
        if action == 2:
            reward = 1
        else:
            reward = 0
    

    rewards= [reward]
    log_probs = [log_prob]
    
    ag.update(rewards,log_probs)

actions = []
treward = 0
for _ in range(500):
    a = int(np.random.choice(2))
    b = int(np.random.choice(2))
    c = int(np.random.choice(2))
    d = int(np.random.choice(2))

    x = torch.Tensor([[a,b,c,d]])
    
    action,log_prob = ag.choose_action(x)

    if a == 1:
        if action == 0:
            reward = 1
        else:   
            reward = 0
    else:
        if action == 2:
            reward = 1
        else:
            reward = 0

    actions.append(action)
        
    treward += reward

print(treward)
            
            