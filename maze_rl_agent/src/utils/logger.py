from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from datetime import datetime
import os

class MazeLogger:
    def __init__(self, log_dir="logs"):
        self.console = Console()
        self.log_dir = log_dir
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{log_dir}/training_{self.current_time}.log"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize progress bars
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        self.training_stats = {
            "episodes": 0,
            "avg_reward": 0,
            "best_reward": float('-inf'),
            "solved_mazes": 0
        }

    def start_training(self, total_episodes):
        self.task = self.progress.add_task("[green]Training Progress", total=total_episodes)
        self.progress.start()
        self.log("Training Started", "info")

    def update_progress(self, episode, reward, steps, success):
        # Update progress bar
        self.progress.update(self.task, advance=1)
        
        # Update stats
        self.training_stats["episodes"] = episode
        self.training_stats["avg_reward"] = (self.training_stats["avg_reward"] * (episode - 1) + reward) / episode
        self.training_stats["best_reward"] = max(self.training_stats["best_reward"], reward)
        if success:
            self.training_stats["solved_mazes"] += 1

        # Log episode details
        self.log(f"Episode {episode}: Reward={reward:.2f}, Steps={steps}, Success={success}", "episode")
        
        # Every 100 episodes, show detailed stats
        if episode % 100 == 0:
            self.show_training_stats()

    def show_training_stats(self):
        table = Table(title="Training Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Episodes", str(self.training_stats["episodes"]))
        table.add_row("Average Reward", f"{self.training_stats['avg_reward']:.2f}")
        table.add_row("Best Reward", f"{self.training_stats['best_reward']:.2f}")
        table.add_row("Solved Mazes", str(self.training_stats["solved_mazes"]))
        self.console.print(table) 

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message with the specified level.
        
        Args:
            message (str): The message to log
            level (str): The log level ('info', 'warning', 'error', etc.)
        """
        if level == "info":
            print(f"[INFO] {message}")
        elif level == "warning":
            print(f"[WARNING] {message}")
        elif level == "error":
            print(f"[ERROR] {message}")
        else:
            print(f"[{level.upper()}] {message}")