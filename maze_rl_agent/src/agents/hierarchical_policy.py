import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalPolicy:
    def __init__(self, maze_size=10, hidden_dim=128):
        self.maze_size = maze_size
        self.hidden_dim = hidden_dim
        
        # High-level policy network (for region segmentation)
        self.region_network = RegionNetwork(
            input_dim=maze_size*maze_size,
            hidden_dim=hidden_dim
        )
        
        # Low-level policy network (for local navigation)
        self.local_network = LocalNetwork(
            input_dim=maze_size*maze_size,
            hidden_dim=hidden_dim,
            action_dim=4
        )
    
    def segment_maze(self, maze_state):
        """Divide maze into regions based on complexity"""
        state_tensor = torch.FloatTensor(maze_state).flatten()
        regions = self.region_network(state_tensor)
        return self._process_regions(regions, maze_state)
    
    def _process_regions(self, regions, maze_state):
        """Convert network output to actual maze regions"""
        regions = regions.reshape(self.maze_size, self.maze_size)
        # Use connected components to identify distinct regions
        labeled_regions = []
        visited = set()
        
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if (i, j) not in visited and maze_state[i, j] == 0:
                    region = self._flood_fill(maze_state, i, j, visited)
                    labeled_regions.append(region)
        
        return labeled_regions
    
    def _flood_fill(self, maze, x, y, visited):
        """Helper method to find connected regions"""
        if (x, y) in visited or not (0 <= x < self.maze_size and 0 <= y < self.maze_size):
            return set()
        if maze[x, y] == 1:  # Wall
            return set()
            
        region = {(x, y)}
        visited.add((x, y))
        
        # Check all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            region.update(self._flood_fill(maze, x + dx, y + dy, visited))
            
        return region

class RegionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class LocalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)