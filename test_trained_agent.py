import gymnasium as gym
import numpy as np
import argparse
import os
from sb3_contrib import QRDQN

from bomber_env_new import BomberEnv # Using the new environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def test_agent(model_path, episodes=5, render=True, deterministic=True):
    """
    Tests a trained agent in the BomberEnv.

    Args:
        model_path (str): Path to the trained model file (.zip).
        episodes (int): Number of episodes to run.
        render (bool): Whether to render the environment.
        deterministic (bool): Whether to use deterministic actions from the model.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return

    render_mode = 'human' if render else None
    # Create a non-vectorized environment first
    raw_env = BomberEnv(render_mode=render_mode)
    # Wrap it in DummyVecEnv for compatibility with VecFrameStack
    env = DummyVecEnv([lambda: raw_env])
    # Apply Frame Stacking (assuming n_stack=4 was used during training)
    env = VecFrameStack(env, n_stack=4)

    try:
        # When loading, SB3 automatically handles the VecEnv wrapper if the model was saved with one.
        # However, the observation space check happens against the env you provide here.
        model = QRDQN.load(model_path, env=env)
        print(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    print(f"\nRunning trained QRDQN agent for {episodes} episodes...")
    print(f"Deterministic actions: {deterministic}\n")

    for episode_num in range(episodes):
        print(f"--- Episode {episode_num + 1} ---")
        obs = env.reset() # VecEnv reset returns only obs
        terminated = False
        truncated = False
        total_episode_reward = 0
        step_num = 0
        
        while not (terminated or truncated):
            step_num += 1
            action, _states = model.predict(obs, deterministic=deterministic)
            
            obs, reward, done, info_vec = env.step(action)
            # Extract info for the single environment
            # For VecEnvs, 'done' is an array, 'terminated' and 'truncated' are in info_vec[0]
            terminated = done[0]
            truncated = info_vec[0].get('TimeLimit.truncated', False) 
            # Ensure info is the dictionary from the actual environment step
            info = info_vec[0]
            total_episode_reward += reward
            
            if render:
                env.render()
            
            print(f"  Step {step_num}: Taking action {action}")
            print(f"    Reward: {reward[0]:.2f}")
            if 'rewards_breakdown' in info and info['rewards_breakdown']:
                print(f"    Rewards Breakdown: {info['rewards_breakdown']}")
            if 'current_walls' in info:
                print(f"    Current Walls: {info['current_walls']}")
            if 'player_alive' in info:
                print(f"    Player Alive: {info['player_alive']}")

        print(f"Episode {episode_num + 1} finished after {step_num} steps.")
        print(f"Total Reward for Episode {episode_num + 1}: {total_episode_reward[0]:.2f}\n")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained Bomberman agent.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.zip file).')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run.')
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable rendering.')
    parser.add_argument('--stochastic', action='store_false', dest='deterministic', help='Use stochastic actions instead of deterministic.')
    
    args = parser.parse_args()
    
    test_agent(model_path=args.model_path, episodes=args.episodes, render=args.render, deterministic=args.deterministic)
