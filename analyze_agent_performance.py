import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from bomber_env_new import BomberEnv # Changed to bomber_env_new
ACTION_BOMB = 5 # From bomber_env_new.py
from collections import defaultdict
import argparse

# Define reward keys from bomber_env_new.py info['rewards_breakdown']
KNOWN_REWARD_KEYS = [
    "strategic_bomb_placement",
    "ineffective_move_penalty",
    "wall_broken",
    "wall_break_survival_bonus",
    "stagnation_penalty",
    "exploration",
    "repetitive_action_still",
    "repetitive_action_bomb",
    "successful_dodge",
    "death_penalty"
]

# Define episode outcome categories based on bomber_env_new.py logic
EPISODE_OUTCOMES = [
    "Agent_Died",
    "Win_Walls_Cleared",
    "Timeout_Max_Steps",
    "Other_Termination" # Fallback for unexpected cases
]

def analyze_performance(model_path, algo, num_episodes=100, env_verbose=0):
    print(f"Analyzing model: {model_path} (algo: {algo}) for {num_episodes} episodes...")

    # Initialize environment
    raw_env = BomberEnv(render_mode=None) # bomber_env_new doesn't use curriculum_stage or verbose in __init__
    env = DummyVecEnv([lambda: raw_env])
    env = VecFrameStack(env, n_stack=4) # Match the n_stack used during training (likely 4 if 8 channels -> 32 channels)
    # It's good practice to wrap with Monitor for SB3, though not strictly necessary for custom analysis if info is rich
    # from stable_baselines3.common.monitor import Monitor
    # env = Monitor(env) # Monitor should wrap the raw_env *before* DummyVecEnv if used for logging episode stats

    # Load the trained model
    if algo == 'dqn':
        ModelClass = DQN
    elif algo == 'qrdqn':
        ModelClass = QRDQN
    elif algo == 'ppo':
        ModelClass = PPO
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    model = ModelClass.load(model_path, env=env)

    # --- Data Aggregators ---
    # For rewards/penalties per source
    total_rewards_by_source = defaultdict(float)
    reward_event_counts = defaultdict(int)

    # For episode outcomes
    outcome_counts = defaultdict(int)

    # For general episode stats
    episode_total_rewards = []
    episode_lengths = []
    episode_walls_broken = []
    episode_bombs_placed = []

    total_steps = 0

    for i_episode in range(num_episodes):
        obs = env.reset() # VecEnv reset returns only obs
        # done flag for the episode loop will be managed by step returns
        current_episode_reward = 0
        current_episode_length = 0
        current_episode_walls_broken = 0
        current_episode_bombs_placed = 0

        # Loop for steps within an episode
        while True:
            action, _states = model.predict(obs, deterministic=True)
            new_obs, rewards, dones, infos = env.step(action)

            obs = new_obs
            reward = rewards[0]
            done = dones[0] # This is True if the episode terminated OR truncated
            info = infos[0]
            # truncated status is usually in info for VecEnvs via TimeLimit wrapper
            truncated = info.get("TimeLimit.truncated", False) 

            current_episode_reward += reward
            current_episode_length += 1
            total_steps +=1

            # Aggregate rewards/penalties from info['rewards_breakdown']
            info_rewards = info.get('rewards_breakdown', {})
            for key, value in info_rewards.items():
                if value != 0: # Only count if there was an actual reward/penalty
                    total_rewards_by_source[key] += value
                    reward_event_counts[key] += 1
                    # episode_specific_rewards[key] += value
            
            # Track bombs placed
            if int(action) == ACTION_BOMB and info.get('action_taken_successfully', False):
                current_episode_bombs_placed += 1

            # Track walls broken this step (from bomber_env_new info dict)
            current_episode_walls_broken += info.get('walls_broken', 0)

            if done:
                episode_total_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_walls_broken.append(current_episode_walls_broken)
                episode_bombs_placed.append(current_episode_bombs_placed)

                # Determine episode outcome
                player_alive_at_end = info.get('player_alive', False)
                current_walls_at_end = info.get("current_walls", -1)
                # 'truncated' is already defined from info.get("TimeLimit.truncated", False)

                if not player_alive_at_end:
                    outcome_counts["Agent_Died"] += 1
                elif current_walls_at_end == 0 and player_alive_at_end:
                    outcome_counts["Win_Walls_Cleared"] += 1
                elif truncated and player_alive_at_end: # Use the 'truncated' flag from TimeLimit.truncated
                    outcome_counts["Timeout_Max_Steps"] += 1
                elif done: # Fallback for other termination reasons if 'done' is true but not covered above
                    outcome_counts["Other_Termination"] += 1
                
                break # End of current episode if using env attributes directly for some stats
                # env.total_blocks_broken_episode = 0 # Example if this was how it's tracked
        
        if (i_episode + 1) % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"  Completed episode {i_episode + 1}/{num_episodes}")

    env.close()

    # --- Report Generation ---
    print("\n--- Agent Performance Analysis ---")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps simulated: {total_steps}")

    # General Stats
    avg_ep_reward = np.mean(episode_total_rewards) if episode_total_rewards else 0
    std_ep_reward = np.std(episode_total_rewards) if episode_total_rewards else 0
    avg_ep_length = np.mean(episode_lengths) if episode_lengths else 0
    avg_walls_broken = np.mean(episode_walls_broken) if episode_walls_broken else 0
    avg_bombs_placed = np.mean(episode_bombs_placed) if episode_bombs_placed else 0

    print("\n--- General Episode Stats ---")
    print(f"Average Episode Reward: {avg_ep_reward:.2f} +/- {std_ep_reward:.2f}")
    print(f"Average Episode Length: {avg_ep_length:.2f} steps")
    print(f"Average Walls Broken per Episode: {avg_walls_broken:.2f}")
    print(f"Average Bombs Placed per Episode: {avg_bombs_placed:.2f}")

    # Reward/Penalty Breakdown
    print("\n--- Reward/Penalty Breakdown (Sum over all episodes) ---")
    sorted_rewards = sorted(total_rewards_by_source.items(), key=lambda item: abs(item[1]), reverse=True)
    for key, total_value in sorted_rewards:
        avg_value_per_occurrence = total_value / reward_event_counts[key] if reward_event_counts[key] > 0 else 0
        occurrences = reward_event_counts[key]
        avg_value_per_episode = total_value / num_episodes
        print(f"  {key+':':<35} Total: {total_value:>10.2f} | Occurrences: {occurrences:>6} | Avg/Occurrence: {avg_value_per_occurrence:>8.2f} | Avg/Episode: {avg_value_per_episode:>8.2f}")
    
    # Check for keys in KNOWN_REWARD_KEYS that didn't appear
    for key in KNOWN_REWARD_KEYS:
        if key not in total_rewards_by_source:
            print(f"  {key+':':<35} Total:      0.00 | Occurrences:      0 | Avg/Occurrence:     0.00 | Avg/Episode:     0.00 (Not observed)")

    # Episode Outcome Breakdown
    print("\n--- Episode Outcome Breakdown ---")
    if sum(outcome_counts.values()) == 0:
        print("  No episode outcomes recorded.")
    else:
        sorted_outcomes = sorted(outcome_counts.items(), key=lambda item: item[0]) # Sort for consistent order
        for outcome, count in sorted_outcomes:
            percentage = (count / num_episodes) * 100
            print(f"  {outcome+':':<25} Count: {count:>5} ({percentage:.1f}% of episodes)")
        # Check for EPISODE_OUTCOMES that didn't appear
        for outcome in EPISODE_OUTCOMES:
            if outcome not in outcome_counts:
                print(f"  {outcome+':':<25} Count:     0 (0.0% of episodes) (Not observed)")

    print("\nAnalysis complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Bomberman RL agent performance with bomber_env_new.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.zip file).')
    parser.add_argument('--algo', type=str, required=True, choices=['dqn', 'qrdqn', 'ppo'], help='Algorithm the model was trained with.')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to simulate for analysis.')
    parser.add_argument('--env-verbose', type=int, default=0, help='Verbosity level for the environment (Note: bomber_env_new may not use this).')

    args = parser.parse_args()
    analyze_performance(args.model_path, args.algo, args.num_episodes, args.env_verbose)
