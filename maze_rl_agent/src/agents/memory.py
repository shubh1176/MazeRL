import torch
import numpy as np

class MemoryMatrix:
    def __init__(self, memory_size=1000, state_dim=100, key_dim=64):
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.key_dim = key_dim
        
        # Initialize memory matrices
        self.keys = torch.zeros(memory_size, key_dim)
        self.values = torch.zeros(memory_size, state_dim)
        self.usage = torch.zeros(memory_size)
        
        # Memory encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, key_dim)
        )
    
    def query(self, state):
        # Encode state to key
        query_key = self.encoder(torch.FloatTensor(state.flatten()))
        
        # Calculate similarity scores
        similarities = torch.cosine_similarity(
            query_key.unsqueeze(0),
            self.keys,
            dim=1
        )
        
        # Get top-k similar memories
        k = 5
        top_k_values, indices = torch.topk(similarities, k)
        
        return self.values[indices]
    
    def update(self, state, value):
        # Find least used memory slot
        min_usage_idx = torch.argmin(self.usage)
        
        # Update memory
        self.keys[min_usage_idx] = self.encoder(
            torch.FloatTensor(state.flatten())
        )
        self.values[min_usage_idx] = torch.FloatTensor(value.flatten())
        self.usage[min_usage_idx] = 1.0
        
        # Decay usage
        self.usage *= 0.99