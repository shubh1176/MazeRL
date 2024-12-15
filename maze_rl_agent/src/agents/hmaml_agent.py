import torch
import torch.nn as nn
import numpy as np
from .memory import MemoryMatrix
from .hierarchical_policy import HierarchicalPolicy
from .meta_learner import MetaLearner

class HMAMLAgent:
    def __init__(self, maze_size=10, memory_size=1000, learning_rate=0.001):
        self.maze_size = maze_size
        self.state_dim = maze_size * maze_size
        
        # Initialize components
        self.memory = MemoryMatrix(
            memory_size=memory_size,
            state_dim=self.state_dim,
            key_dim=64
        )
        
        self.hierarchical_policy = HierarchicalPolicy(
            maze_size=maze_size,
            hidden_dim=128
        )
        
        self.meta_learner = MetaLearner(
            state_dim=self.state_dim,
            hidden_dim=128,
            meta_lr=learning_rate
        )
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def select_action(self, state):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).flatten()
        
        # Query memory for similar experiences
        similar_patterns = self.memory.query(state)
        
        # Get maze regions from hierarchical policy
        regions = self.hierarchical_policy.segment_maze(state)
        
        # Get adapted policy from meta-learner
        adapted_policy = self.meta_learner.adapt(state, similar_patterns)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 4)  # Random action
        else:
            # Get action probabilities from adapted policy
            with torch.no_grad():
                action_probs = adapted_policy(state_tensor)
                action = torch.argmax(action_probs).item()
        
        return action
    
    def train(self, state, action, reward, next_state, done):
        # Update memory
        self.memory.update(state, next_state)
        
        # Get regions for current and next state
        current_regions = self.hierarchical_policy.segment_maze(state)
        next_regions = self.hierarchical_policy.segment_maze(next_state)
        
        # Train meta-learner
        loss = self._compute_meta_loss(state, action, reward, next_state, done)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def _compute_meta_loss(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).flatten()
        next_state_tensor = torch.FloatTensor(next_state).flatten()
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])
        
        # Query memory for both states
        similar_patterns = self.memory.query(state)
        next_similar_patterns = self.memory.query(next_state)
        
        # Get adapted policies
        current_policy = self.meta_learner.adapt(state, similar_patterns)
        next_policy = self.meta_learner.adapt(next_state, next_similar_patterns)
        
        # Compute temporal difference error
        current_value = current_policy(state_tensor)[action]
        next_value = torch.max(next_policy(next_state_tensor))
        expected_value = reward_tensor + (1 - done_tensor) * 0.99 * next_value
        
        # Compute loss
        loss = nn.MSELoss()(current_value, expected_value.detach())
        
        return loss.item()