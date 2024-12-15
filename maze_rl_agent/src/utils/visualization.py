import pygame
import numpy as np

class MazeVisualizer:
    def __init__(self, maze_size, cell_size=40):
        self.maze_size = maze_size
        self.cell_size = cell_size
        self.width = maze_size * cell_size
        self.height = maze_size * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("HMAML Maze Solver")
        
        # Colors
        self.colors = {
            'wall': (0, 0, 0),      # Black
            'path': (255, 255, 255), # White
            'agent': (0, 255, 0),    # Green
            'goal': (255, 0, 0),     # Red
            'region': (0, 0, 255, 50) # Semi-transparent blue
        }
    
    def render(self, maze_state, agent_pos, regions=None):
        self.screen.fill(self.colors['path'])
        
        # Draw maze
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                x = j * self.cell_size
                y = i * self.cell_size
                
                if maze_state[i][j] == 1:  # Wall
                    pygame.draw.rect(
                        self.screen,
                        self.colors['wall'],
                        (x, y, self.cell_size, self.cell_size)
                    )
        
        # Draw regions if provided
        if regions:
            for region in regions:
                surface = pygame.Surface(
                    (self.width, self.height),
                    pygame.SRCALPHA
                )
                for cell in region:
                    x = cell[1] * self.cell_size
                    y = cell[0] * self.cell_size
                    pygame.draw.rect(
                        surface,
                        self.colors['region'],
                        (x, y, self.cell_size, self.cell_size)
                    )
                self.screen.blit(surface, (0, 0))
        
        # Draw agent
        agent_x = agent_pos[1] * self.cell_size + self.cell_size // 2
        agent_y = agent_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(
            self.screen,
            self.colors['agent'],
            (agent_x, agent_y),
            self.cell_size // 3
        )
        
        # Draw goal
        goal_x = (self.maze_size - 1) * self.cell_size + self.cell_size // 2
        goal_y = (self.maze_size - 1) * self.cell_size + self.cell_size // 2
        pygame.draw.circle(
            self.screen,
            self.colors['goal'],
            (goal_x, goal_y),
            self.cell_size // 3
        )
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()