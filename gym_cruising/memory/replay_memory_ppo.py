from collections import namedtuple, deque
import random
import torch
import numpy as np

Transition = namedtuple('Transition', ('states', 'actions', 'log_probs', 'rewards', 'terminated', 'values', 'advantages', 'returns'))


class ReplayMemoryPPO(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_tensor_batch(self, device="cpu"):
        if len(self.memory) == 0:
            raise ValueError("Replay buffer is empty.")

        batch = Transition(*zip(*self.memory))

        # Converti stati (già con padding)
        states = torch.tensor(np.array(batch.states), dtype=torch.float32).to(device)

        # Converti azioni (lista di [UAV, 2]) → tensor (batch, UAV, 2)
        actions = torch.tensor(np.array(batch.actions), dtype=torch.float32).to(device)

        # Converti log_probs (lista di [UAV]) → tensor (batch, UAV)
        log_probs = torch.tensor(np.array(batch.log_probs), dtype=torch.float32).to(device)

        # Converti rewards (lista di [UAV]) → tensor (batch, UAV)
        rewards = torch.tensor(np.array(batch.rewards), dtype=torch.float32).to(device)

        # Converti dones (1D list → tensor (batch, 1))
        terminated = torch.tensor(np.array(batch.terminated), dtype=torch.float32).unsqueeze(1).to(device)

        # Converti values (lista di [UAV]) → tensor (batch, UAV)
        values = torch.tensor(np.array(batch.values), dtype=torch.float32).to(device)

        # Converti advantages (lista di [UAV]) → tensor (batch, UAV)
        advantages = torch.tensor(np.array(batch.advantages), dtype=torch.float32).to(device)

        # Converti returns (lista di [UAV]) → tensor (batch, UAV)
        returns = torch.tensor(np.array(batch.returns), dtype=torch.float32).to(device)

        return states, actions, log_probs, rewards, terminated, values, advantages, returns
    
    def get_minibatch(self, mb_inds):
        batch = [self.memory[i] for i in mb_inds]
        return batch 
    
    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
