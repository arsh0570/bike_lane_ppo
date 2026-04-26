import os
import sys
import time
import argparse
import pandas as pd
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# Optional: wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add the project root to path so we can find ebike_city_tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ebike_city_tools.graph_utils import load_lane_graph, keep_only_the_largest_connected_component
from ebike_city_tools.optimize.rl_env_optimise import BikeNetworkEnv


class WandbCallback(BaseCallback):
    """Custom callback for logging metrics to wandb during PPO training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("done"):
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            "episode_reward": info["episode"]["r"],
                            "episode_length": info["episode"]["l"],
                            "timestep": self.num_timesteps
                        })
                    
                    if self.verbose > 0:
                        print(f"Timestep {self.num_timesteps}: "
                              f"Episode Reward = {info['episode']['r']:.2f}, "
                              f"Length = {info['episode']['l']}")
        
        if self.num_timesteps % 100 == 0 and WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {"timestep": self.num_timesteps}
            for key, value in self.model.logger.name_to_value.items():
                if isinstance(value, (int, float)):
                    log_dict[key] = value
            wandb.log(log_dict)
        
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for bike lane reallocation")
    parser.add_argument("--instance", type=str, default="affoltern",
                        help="Name of the instance (subfolder under data/)")
    parser.add_argument("--data_root", type=str, default="..",
                        help="Root directory containing the data/ folder")
    parser.add_argument("--total_timesteps", type=int, default=1500,
                        help="Total number of timesteps for training")
    parser.add_argument("--learning_rate", type=float, default=0.0003,
                        help="Learning rate for PPO")
    parser.add_argument("--policy", type=str, default="MlpPolicy",
                        help="Policy network type (MlpPolicy, etc.)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--output_dir", type=str, default="../models",
                        help="Directory where the trained model will be saved")
    parser.add_argument("--wandb_project", type=str, default="bike-lane-rl",
                        help="W&B project name (if wandb is installed)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Custom run name for W&B (auto-generated if None)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging even if available")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level for PPO (0,1)")
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Determine data path
    data_path = os.path.join(args.data_root, "data", args.instance)
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist. "
              f"Make sure you are running from the scripts/ folder or adjust --data_root.")
        sys.exit(1)
    
    # Initialize wandb if available and not disabled
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"ppo_{args.instance}_{args.total_timesteps}steps_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "algorithm": "PPO",
                "instance": args.instance,
                "total_timesteps": args.total_timesteps,
                "learning_rate": args.learning_rate,
                "policy": args.policy,
                "device": args.device,
            }
        )
        print(f"Logging to W&B: {args.wandb_project}/{run_name}")
    else:
        if not WANDB_AVAILABLE and not args.no_wandb:
            print("wandb not installed. Install with: pip install wandb")
        print("W&B logging disabled.")
    
    # 1. Load Data
    print(f"\n--- Loading Data for {args.instance} from {data_path} ---")
    G_lane = load_lane_graph(data_path)
    G_lane = keep_only_the_largest_connected_component(G_lane)
    
    od = pd.read_csv(os.path.join(data_path, "od_matrix.csv"))
    od = od[od["s"] != od["t"]]
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
    
    print(f"Graph: {G_lane.number_of_nodes()} nodes, {G_lane.number_of_edges()} edges")
    print(f"OD Matrix: {len(od)} rows")
    
    # 2. Initialize Environment
    env = BikeNetworkEnv(G_lane, od)
    print(f"Observation Space Shape: {env.observation_space.shape}")
    print("Checking environment...")
    check_env(env)
    
    # 3. Train Agent
    print(f"\nTraining starting ({args.total_timesteps} steps)...")
    model = PPO(
        args.policy,
        env,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    callback = WandbCallback(verbose=args.verbose) if use_wandb else None
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    
    # 4. Save Model with Timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"bike_lane_agent_{args.instance}_{timestamp}"
    save_path = os.path.join(args.output_dir, model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(save_path)
    
    print(f"\n--- Training Complete ---")
    print(f"Model saved to: {save_path}.zip")
    
    if use_wandb:
        wandb.log({"final_timestep": args.total_timesteps, "model_saved": save_path})
        wandb.finish()


if __name__ == "__main__":
    train()