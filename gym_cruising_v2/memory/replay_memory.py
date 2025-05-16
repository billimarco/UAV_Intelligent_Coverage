from collections import namedtuple, deque
import random
import torch
import numpy as np

Transition = namedtuple('Transition', ('states', 'actions', 'next_states', 'rewards', 'terminated'))


class ReplayMemory(object):

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

        # Convert to PyTorch tensors
        states = torch.tensor(np.array(batch.states), dtype=torch.float32).to(device)

        # Actions: list of tuples -> tuple of tensors
        actions = tuple(
            torch.tensor([a[i] for a in batch.actions], dtype=torch.float32).to(device)
            for i in range(len(batch.actions[0]))
        )

        next_states = torch.tensor(np.array(batch.next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(batch.rewards), dtype=torch.float32).unsqueeze(1).to(device)
        terminated = torch.tensor(np.array(batch.terminated), dtype=torch.float32).unsqueeze(1).to(device)

        return states, actions, next_states, rewards, terminated
    
    def get_minibatch(self, mb_inds):
        batch = [self.memory[i] for i in mb_inds]
        return batch 
    
    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
