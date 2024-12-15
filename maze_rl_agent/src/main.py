import numpy as np
import torch
import pygame as pg
from environment.maze_env import MazeEnvironment
from agents.hmaml_agent import HMAMLAgent
from utils.logger import MazeLogger
import matplotlib.pyplot as plt
from collections import deque

class MazeSolver:
    def __init__(self, maze_size=10, memory_size=1000, max_episodes=1000):
        # Initialize Pygame (based on grid.py example)
        pg.init()
        self.screen_size = 800
        self.cell_size = self.screen_size // maze_size
        self.screen = pg.display.set_mode((self.screen_size, self.screen_size))
        pg.display.set_caption("Maze Solver RL")
        self.clock = pg.time.Clock()
        
        # Colors for visualization
        self.colors = {
            'background': (0, 0, 0),
            'wall': (40, 40, 40),
            'agent': (255, 255, 255),
            'goal': (0, 255, 0),
            'path': (0, 0, 255, 128),
            'region': (255, 0, 0, 64)
        }
        
        # Initialize environment and agent
        self.env = MazeEnvironment(size=maze_size)
        self.agent = HMAMLAgent(maze_size=maze_size, memory_size=memory_size)
        self.max_episodes = max_episodes
        self.logger = MazeLogger()
        
        # Training metrics
        self.rewards_history = []
        self.steps_history = []
        self.success_rate = deque(maxlen=100)

    def render_maze(self, state, regions=None):
        self.screen.fill(self.colors['background'])
        
        # Draw maze grid
        for row in range(self.env.size):
            for col in range(self.env.size):
                rect = (col * self.cell_size, row * self.cell_size, 
                       self.cell_size, self.cell_size)
                
                if self.env.maze[row][col] == 1:
                    pg.draw.rect(self.screen, self.colors['wall'], rect)
        
        # Draw goal
        goal_rect = (self.env.goal_pos[1] * self.cell_size, 
                    self.env.goal_pos[0] * self.cell_size,
                    self.cell_size, self.cell_size)
        pg.draw.rect(self.screen, self.colors['goal'], goal_rect)
        
        # Draw agent with error handling
        try:
            if hasattr(self.env, 'agent_pos') and self.env.agent_pos is not None:
                agent_pos = (
                    int(self.env.agent_pos[1] * self.cell_size + self.cell_size/2),
                    int(self.env.agent_pos[0] * self.cell_size + self.cell_size/2)
                )
                pg.draw.circle(self.screen, self.colors['agent'], agent_pos, self.cell_size//3)
            else:
                print("Warning: Agent position not available")
        except Exception as e:
            print(f"Error drawing agent: {e}")
            return
        
        pg.display.flip()
        self.clock.tick(60)

    def log_progress(self, episode):
        """Log training progress"""
        avg_reward = np.mean(self.rewards_history[-10:])
        avg_steps = np.mean(self.steps_history[-10:])
        success_rate = np.mean(list(self.success_rate)[-10:])
        
        self.logger.update_progress(
            episode=episode + 1,
            reward=avg_reward,
            steps=avg_steps,
            success=(success_rate > 0)
        )

    def train(self, render=True):
        self.logger.start_training(self.max_episodes)
        
        try:
            for episode in range(self.max_episodes):
                state, _ = self.env.reset()
                print(f"Initial state: {state}, type: {type(state)}")  # Debug print
                
                episode_reward = 0
                steps = 0
                done = False
                
                # Get initial regions
                regions = self.agent.hierarchical_policy.segment_maze(state) if render else None
                
                while not done:
                    # Handle Pygame events less frequently
                    if steps % 5 == 0:  # Only check events every 5 steps
                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                raise KeyboardInterrupt
                            if event.type == pg.KEYDOWN:
                                if event.key == pg.K_q:
                                    raise KeyboardInterrupt
                
                    # Render less frequently
                    if render and steps % 2 == 0:  # Render every other step
                        self.render_maze(state, regions)
                
                    # Agent interaction
                    action = self.agent.select_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                
                    # Train agent
                    loss = self.agent.train(state, action, reward, next_state, done)
                
                    # Update metrics
                    episode_reward += reward
                    steps += 1
                    state = next_state
                
                    # Update regions less frequently
                    if render and (done or steps % 20 == 0):  # Changed from 10 to 20
                        regions = self.agent.hierarchical_policy.segment_maze(state)
                
                # Update episode metrics
                self.rewards_history.append(episode_reward)
                self.steps_history.append(steps)
                self.success_rate.append(1 if episode_reward > 0 else 0)
                
                # Log progress every 10 episodes
                if (episode + 1) % 10 == 0:
                    self.log_progress(episode)
                
        except KeyboardInterrupt:
            self.logger.log("Training interrupted by user", "warning")
        finally:
            pg.quit()
            self.plot_training_results()

    def plot_training_results(self):
        """Plot training metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rewards
        ax1.plot(self.rewards_history)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        # Plot steps
        ax2.plot(self.steps_history)
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        
        # Plot success rate
        success_rates = [np.mean(list(self.success_rate)[max(0, i-100):i+1])
                        for i in range(len(self.rewards_history))]
        ax3.plot(success_rates)
        ax3.set_title('Success Rate (Moving Average)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
    
    def evaluate(self, num_episodes=10, render=True):
        """Evaluate the trained agent"""
        print("\nStarting evaluation...")
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            
            while not done:
                if render:
                    self.render_maze(
                        state,
                        self.agent.hierarchical_policy.segment_maze(state)
                    )
                    pg.time.wait(100)
                
                action = self.agent.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                steps += 1
                
                if done and reward > 0:
                    success_count += 1
            
            print(f"Episode {episode + 1}: {'Success' if reward > 0 else 'Failure'} in {steps} steps")
        
        print(f"\nEvaluation Success Rate: {success_count/num_episodes:.2%}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create and train the maze solver
    solver = MazeSolver(
        maze_size=10,
        memory_size=1000,
        max_episodes=500
    )
    
    # Train the agent
    solver.train(render=True)
    
    # Evaluate the trained agent
    solver.evaluate(num_episodes=10, render=True)