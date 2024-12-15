import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MazeEnvironment(gym.Env):
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size, size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        
        # Initialize maze
        self.maze = np.zeros((size, size))
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.current_pos = self.start_pos
        self.agent_pos = None
        
        self._generate_maze()
    
    def _generate_maze(self):
        # Simple maze generation for now
        # Add walls randomly
        self.maze = np.random.choice(
            [0, 1], 
            size=(self.size, self.size), 
            p=[0.7, 0.3]
        )
        # Ensure start and goal positions are free
        self.maze[self.start_pos] = 0
        self.maze[self.goal_pos] = 0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        self.agent_pos = None
        return self._get_observation(), {}
    
    def _get_observation(self):
        obs = self.maze.copy()
        obs[self.current_pos] = 0.5  # Mark current position
        return obs
    
    def step(self, action):
        # Action mapping
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Calculate new position
        new_pos = (
            self.current_pos[0] + moves[action][0],
            self.current_pos[1] + moves[action][1]
        )
        
        # Check if move is valid
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            self.maze[new_pos] == 0):
            self.current_pos = new_pos
            self.agent_pos = new_pos
            
            # Check if goal reached
            if self.current_pos == self.goal_pos:
                return self._get_observation(), 10.0, True, False, {}
            
            return self._get_observation(), -0.1, False, False, {}
        
        return self._get_observation(), -1.0, False, False, {}