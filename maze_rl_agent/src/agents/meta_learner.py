import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MetaLearner:
    def __init__(self, state_dim=100, hidden_dim=128, meta_lr=0.01):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.meta_lr = meta_lr
        
        # Meta-learning network
        self.meta_network = MetaNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim
        )
        
        self.optimizer = torch.optim.Adam(
            self.meta_network.parameters(),
            lr=meta_lr
        )
    
    def adapt(self, region, similar_patterns):
        """Adapt policy for current region based on similar past experiences"""
        region_tensor = torch.FloatTensor(region.flatten())
        patterns_tensor = torch.FloatTensor(similar_patterns)
        
        # Get adaptation parameters from meta-network
        adaptation = self.meta_network(region_tensor, patterns_tensor)
        
        return self._apply_adaptation(adaptation, region)
    
    def _apply_adaptation(self, adaptation, region):
        """Apply the adaptation to create a region-specific policy"""
        adapted_policy = AdaptedPolicy(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            adaptation=adaptation
        )
        return adapted_policy

class MetaNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.adaptation_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim)
        )
    
    def forward(self, current_state, similar_patterns):
        # Encode current state
        current_encoding = self.encoder(current_state)
        
        # Encode similar patterns
        pattern_encoding = self.encoder(similar_patterns.mean(dim=0))
        
        # Generate adaptation parameters
        combined = torch.cat([current_encoding, pattern_encoding], dim=-1)
        adaptation = self.adaptation_generator(combined)
        
        # Reshape to match target layer dimensions [4, hidden_dim]
        adaptation = adaptation.reshape(4, -1)
        
        return adaptation

class AdaptedPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, adaptation):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 actions: up, right, down, left
        )
        
        # Apply adaptation parameters
        self._adapt_network(adaptation)
    
    def _adapt_network(self, adaptation):
        """Apply adaptation parameters to the last layer weights"""
        last_layer = self.network[-1].weight  # Shape is [4, hidden_dim]
        # Ensure adaptation has same shape
        assert last_layer.shape == adaptation.shape, f"Shape mismatch: {last_layer.shape} vs {adaptation.shape}"
        self.network[-1].weight = nn.Parameter(last_layer + adaptation)
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)