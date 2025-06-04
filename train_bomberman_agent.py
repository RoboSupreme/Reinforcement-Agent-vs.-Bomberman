import os
import argparse
import time
import torch.nn as nn
import torch as th
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from bomber_env_new import BomberEnv # Using the new environment

# Custom CNN feature extractor for the small Bomberman board
class SmallBoardCNN(BaseFeaturesExtractor):
    """CNN feature extractor for small board games like the 7x11 Bomberman board."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[2]  # Channels last
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            # Permute from (B, H, W, C) to (B, C, H, W) for PyTorch CNN
            sample_obs_permuted = sample_obs.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_obs_permuted).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Permute from (B, H, W, C) to (B, C, H, W) for PyTorch CNN
        observations_permuted = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations_permuted))

def make_env(opponent_style="still", log_dir=None, rank=0, seed=0, info_keywords_to_log=()):
    """
    Utility function for multiprocessed env.
    :param opponent_style: (str) Type of opponent for BomberEnv
    :param log_dir: (str) Firectory for Monitor logs
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param info_keywords_to_log: (tuple) extra info keywords to log
    """
    def _init():
        env = BomberEnv() # New env does not take opponent or verbose
        # Important: wrap the environment with Monitor to log rewards and other info
        # The info_keywords argument tells Monitor which keys from the info dict to log
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            monitor_path = os.path.join(log_dir, str(rank))
        else:
            monitor_path = None
        env = Monitor(env, filename=monitor_path, info_keywords=info_keywords_to_log)
        env.reset(seed=seed + rank)
        return env
    return _init

# Custom callback to log 'current_walls' at specified step intervals
class StepwiseWallLoggerCallback(BaseCallback):
    def __init__(self, log_frequency: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency

    def _on_step(self) -> bool:
        if self.n_calls % self.log_frequency == 0:
            # For DummyVecEnv, infos is a list of dicts, one for each env
            # We assume a single environment here (index 0)
            if self.locals.get('infos') and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'current_walls' in info:
                    current_walls = info['current_walls']
                    print(f"Step: {self.num_timesteps}, Current Walls: {current_walls}")
        return True

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Bomberman agent')
    parser.add_argument('--algo', type=str, default='qrdqn', choices=['dqn', 'qrdqn', 'ppo'], help='RL algorithm to use')
    parser.add_argument('--total-timesteps', type=int, default=500000, help='Total timesteps for training')
    parser.add_argument('--opponent-style', type=str, default='still', help='Opponent style for training (still, random)')
    parser.add_argument('--eval-opponent-style', type=str, default='random', help='Opponent style for evaluation')
    parser.add_argument('--log-dir', type=str, default='logs/bomberman_agent_logs', help='Directory to save Monitor logs')
    parser.add_argument('--model-dir', type=str, default='models/bomberman_agent_models', help='Directory to save trained models')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size (for DQN/QRDQN)')
    parser.add_argument('--learning-starts', type=int, default=10000, help='How many steps to collect before learning starts')
    parser.add_argument('--batch-size', type=int, default=64, help='Minibatch size')
    parser.add_argument('--target-update-interval', type=int, default=1000, help='Frequency of target network update (for DQN/QRDQN)')
    parser.add_argument('--exploration-fraction', type=float, default=0.3, help='Fraction of training to explore (for DQN/QRDQN)')
    parser.add_argument('--exploration-final-eps', type=float, default=0.05, help='Final exploration probability (for DQN/QRDQN)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--n-stack', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use for training')
    parser.add_argument('--save-freq', type=int, default=50000, help='Save model checkpoint every N steps')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluate model every N steps')
    parser.add_argument('--load-path', type=str, default=None, help='Path to a pre-trained model to continue training')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Create a unique subdirectory for this training run's logs and models
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    current_log_dir = os.path.join(args.log_dir, f"{args.algo}_{timestamp}")
    current_model_dir = os.path.join(args.model_dir, f"{args.algo}_{timestamp}")
    os.makedirs(current_log_dir, exist_ok=True)
    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(os.path.join(current_model_dir, "best"), exist_ok=True)

    # Keywords from the 'info' dictionary returned by env.step() to log with Monitor
    # BomberEnv should return info['blocks'] for this to work.
    info_keywords_to_log = ('current_walls', 'walls_broken')

    # Create training environment
    train_env = DummyVecEnv([make_env(opponent_style=args.opponent_style, 
                                      log_dir=current_log_dir, 
                                      rank=0, 
                                      info_keywords_to_log=info_keywords_to_log)])
    train_env = VecFrameStack(train_env, n_stack=args.n_stack)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(opponent_style=args.eval_opponent_style, 
                                     log_dir=os.path.join(current_log_dir, "eval"), 
                                     rank=100, # Use a different rank/seed base for eval env
                                     info_keywords_to_log=info_keywords_to_log)])
    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)

    # Policy kwargs for the CNN feature extractor and network architecture
    policy_kwargs = dict(
        features_extractor_class=SmallBoardCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],       # For DQN/QRDQN, this is the MLP layers after the CNN
        normalize_images=False # Crucial: BomberEnv output is already [0,1] float32 planes
    )

    # Select algorithm and instantiate or load model
    model_class_map = {
        'dqn': DQN,
        'qrdqn': QRDQN,
        'ppo': PPO
    }
    ModelClass = model_class_map.get(args.algo)
    if ModelClass is None:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # Prepare policy_kwargs, potentially adjusting for PPO
    current_policy_kwargs = policy_kwargs.copy()
    if args.algo == 'ppo':
        current_policy_kwargs.pop('normalize_images', None)

    if args.load_path:
        print(f"Loading model from: {args.load_path}")
        load_kwargs = {
            "env": train_env,
            "tensorboard_log": current_log_dir,
            "device": args.device,
            # "policy_kwargs": current_policy_kwargs, # Removed to use stored policy_kwargs
            "learning_rate": args.lr,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "verbose": 1,
            "target_update_interval": args.target_update_interval,
            "exploration_fraction": args.exploration_fraction,
            "exploration_final_eps": args.exploration_final_eps,
            "n_steps": 2048  # Default for PPO, make it an arg if it needs to change
        }
        # For off-policy algorithms like DQN/QRDQN, some args might not be needed or should be handled carefully when loading
        # We'll let .load() use the saved hyperparameters primarily, but override some for the new session
        if args.algo in ['dqn', 'qrdqn']:
            # These are generally safe to update for a continued session
            load_params_for_off_policy = {
                "learning_rate": args.lr,
                "buffer_size": args.buffer_size, # Can be adjusted if needed
                "learning_starts": args.learning_starts, # Might need adjustment if buffer is new/different
                "gamma": args.gamma,
                "target_update_interval": args.target_update_interval,
                "exploration_fraction": args.exploration_fraction,
                "exploration_final_eps": args.exploration_final_eps,
            }
            # Filter out keys from load_kwargs that are not in load_params_for_off_policy for these algos
            filtered_load_kwargs = {k: v for k, v in load_kwargs.items() if k in load_params_for_off_policy or k in ['env', 'tensorboard_log', 'device', 'verbose']}
            model = ModelClass.load(args.load_path, **filtered_load_kwargs)
        elif args.algo == 'ppo':
            # PPO might have different considerations, for now, pass most relevant
            load_params_for_ppo = {
                "learning_rate": args.lr,
                "gamma": args.gamma,
                "n_steps": 2048, # Or args.n_steps if made an arg
                "batch_size": args.batch_size
            }
            filtered_load_kwargs = {k: v for k, v in load_kwargs.items() if k in load_params_for_ppo or k in ['env', 'tensorboard_log', 'device', 'verbose']}
            model = ModelClass.load(args.load_path, **filtered_load_kwargs)
        else:
            model = ModelClass.load(args.load_path, **load_kwargs) # Fallback, though should be covered
        print(f"Continuing training for {args.algo.upper()} agent...")
    else:
        print(f"Starting new training for {args.algo.upper()} agent...")
        if args.algo == 'dqn':
            model = DQN('CnnPolicy', train_env, policy_kwargs=current_policy_kwargs,
                          learning_rate=args.lr, buffer_size=args.buffer_size,
                          learning_starts=args.learning_starts, batch_size=args.batch_size,
                          gamma=args.gamma, target_update_interval=args.target_update_interval,
                          exploration_fraction=args.exploration_fraction, 
                          exploration_final_eps=args.exploration_final_eps,
                          verbose=1, tensorboard_log=current_log_dir, device=args.device)
        elif args.algo == 'qrdqn':
            model = QRDQN('CnnPolicy', train_env, policy_kwargs=current_policy_kwargs,
                            learning_rate=args.lr, buffer_size=args.buffer_size,
                            learning_starts=args.learning_starts, batch_size=args.batch_size,
                            gamma=args.gamma, target_update_interval=args.target_update_interval,
                            exploration_fraction=args.exploration_fraction, 
                            exploration_final_eps=args.exploration_final_eps,
                            verbose=1, tensorboard_log=current_log_dir, device=args.device)
        elif args.algo == 'ppo':
            model = PPO('CnnPolicy', train_env, policy_kwargs=current_policy_kwargs,
                        learning_rate=args.lr, n_steps=2048, batch_size=args.batch_size,
                        gamma=args.gamma, verbose=1, tensorboard_log=current_log_dir, device=args.device)
    
    print(f"Logs will be saved to: {current_log_dir}")
    print(f"Models will be saved to: {current_model_dir}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, 
                                             save_path=current_model_dir, 
                                             name_prefix=f"{args.algo}_bomber")
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(current_model_dir, 'best'),
                                 log_path=os.path.join(current_log_dir, 'eval_results'),
                                 eval_freq=args.eval_freq, deterministic=True, render=False)

    # Custom callback for step-wise logging
    step_logger_callback = StepwiseWallLoggerCallback(log_frequency=10)
    callbacks = [checkpoint_callback, eval_callback, step_logger_callback]

    # Train the agent
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=(args.load_path is None))

    # Save the final model
    final_model_path = os.path.join(current_model_dir, f"{args.algo}_bomber_final.zip")
    model.save(final_model_path)

    print("--- Training Complete ---")
    print(f"Final model saved to: {final_model_path}")
    print(f"Tensorboard logs: tensorboard --logdir {current_log_dir}")
