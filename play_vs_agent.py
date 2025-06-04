import argparse
import numpy as np
import cv2 # cv2 is used by bomber_env_new for input when render_mode='human'
import time
import pygame
import os
from collections import deque

from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from bomber_env_new import BomberEnv # Assuming HUMAN_PLAYER_KEY_MAP is not explicitly needed here as env handles it

def parse_args():
    parser = argparse.ArgumentParser(description='Play Bomberman against a trained agent.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model (.zip file).')
    parser.add_argument('--algo', type=str, required=True, choices=['dqn', 'qrdqn', 'ppo'],
                        help='Algorithm the model was trained with (dqn, qrdqn, ppo).')
    parser.add_argument('--n-stack', type=int, default=4, help='Number of frames stacked during training.')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions for the agent.')
    parser.add_argument('--no-deterministic', action='store_false', dest='deterministic', help='Use stochastic actions for the agent.')
    parser.set_defaults(deterministic=True)
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return

    print("--- Human vs. AI Bomberman ---")
    print("You are Player 2 (Blue, starts bottom-right).")
    print("AI is Player 1 (Red, starts top-left).")
    print("Controls (Player 2 - focus on the game window for input):")
    print("  W: Up")
    print("  A: Left")
    print("  S: Down")
    print("  D: Right")
    print("  Space: Place Bomb")
    print("  E: Stand Still")
    print("  Q: Quit Game")
    print("-----------------------------------\n")

    # --- Environment Setup --- 
    # Main environment for gameplay.
    # player_controlled='player2' means env.step() expects AI (player1) action,
    # and human (player2) action is read via get_human_input() internally by env.step().
    # opponent_player_num=1 means the AI is player 1.
    game_env = BomberEnv(render_mode='human', player_controlled='player2', opponent_player_num=1)

    # --- Load Model --- 
    # Create a temporary environment that matches the training setup for model loading.
    # This ensures observation space compatibility during model.load().
    def _create_temp_env():
        _env = BomberEnv(render_mode=None) # No rendering for this temp env
        return _env

    temp_vec_env = DummyVecEnv([_create_temp_env])
    # Important: Use the same n_stack as during training for the model to load correctly.
    stacked_temp_env = VecFrameStack(temp_vec_env, n_stack=args.n_stack)

    print(f"Loading model ({args.algo.upper()}) from: {args.model_path}")
    try:
        if args.algo == 'dqn':
            model = DQN.load(args.model_path, env=stacked_temp_env)
        elif args.algo == 'ppo':
            model = PPO.load(args.model_path, env=stacked_temp_env)
        elif args.algo == 'qrdqn':
            model = QRDQN.load(args.model_path, env=stacked_temp_env)
        else:
            # Should be caught by argparse choices, but good practice.
            raise ValueError(f"Unsupported algorithm: {args.algo}")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        stacked_temp_env.close()
        game_env.close()
        return
    finally:
        # Close the temporary environment once the model is loaded or if loading fails.
        stacked_temp_env.close()

    # Initialize Pygame display if in human mode
    if game_env.render_mode == 'human':
        pygame.init()
        clock = pygame.time.Clock() # Initialize the clock
        # screen = pygame.display.set_mode((game_env.width * game_env.tile_size, game_env.height * game_env.tile_size))
        # pygame.display.set_caption("Bomberman - Human vs AI")
        # The environment's render method will now handle display initialization and updates.

    # --- Gameplay Loop --- 
    obs_deque = deque(maxlen=args.n_stack)
    # The observation from reset is for the AI (Player 1)
    raw_obs_p1, info = game_env.reset() 

    # Initialize deque with the first observation repeated n_stack times
    for _ in range(args.n_stack):
        obs_deque.append(raw_obs_p1)

    terminated = False
    truncated = False
    episode_reward_player1 = 0 # AI's reward
    episode_reward_player2 = 0 # Human's reward (from info)
    step_count = 0

    # ---- START DEBUG BLOCK: Print key map ----
    print(f"[DEBUG_INIT] game_env.human_key_map (raw): {game_env.human_key_map}")
    try:
        readable_key_map = {pygame.key.name(k): v for k, v in game_env.human_key_map.items()}
        print(f"[DEBUG_INIT] game_env.human_key_map (readable): {readable_key_map}")
    except Exception as e:
        print(f"[DEBUG_INIT] Error making key_map readable: {e}")
    # ---- END DEBUG BLOCK ----

    try:
        while not (terminated or truncated):
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            game_env.render() # Render the current state

            # Prepare AI's observation (stack frames)
            # The deque already contains the latest n_stack frames for Player 1
            # Concatenate along the last axis (channels) for SB3
            stacked_observation_ai = np.concatenate(list(obs_deque), axis=-1)
            # Add batch dimension for model.predict
            stacked_observation_ai_batch = np.expand_dims(stacked_observation_ai, axis=0)

            # --- Pygame event pumping and Human Player (P2) Input Handling ---
            p2_action_name_this_tick = "Still" # Default action for P2 this tick

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[MAIN_LOOP] Pygame QUIT event detected. Exiting...")
                    terminated = True 
                    break # Break from event loop
                if event.type == pygame.KEYDOWN:
                    if event.key == game_env.quit_key:
                        print(f"[MAIN_LOOP] Game quit key ({pygame.key.name(game_env.quit_key)}) detected. Exiting...")
                        terminated = True
                        break # Break from event loop
                    
                    # Check for P2 action keys if P2 is alive
                    if game_env.human_player_object and game_env.human_player_object.alive:
                        # game_env.human_key_map is {pygame_key_code: "ActionNameString"}
                        for key_code_from_map, action_name_from_map in game_env.human_key_map.items():
                            if event.key == key_code_from_map:
                                p2_action_name_this_tick = action_name_from_map
                                print(f"[DEBUG_P2_INPUT] Event-based P2 action: {p2_action_name_this_tick} from key {pygame.key.name(event.key)}")
                                # Break from iterating human_key_map; current event handled for P2
                                break 
                        # If an action key for P2 was found and processed from this event,
                        # we don't need to check other keys in human_key_map for THIS event.
                        # However, other events in the queue might still be processed in this same game tick's event loop.

            if terminated: # If quit event was processed in the event loop, break before P2/AI turn
                break

            # Human player's turn (P2) - Apply the action determined from events
            p2_object = game_env.human_player_object

            # ---- START DEBUG BLOCK: P2 Object Status ----
            if p2_object:
                print(f"[DEBUG_P2_INPUT] P2 Object ID: {id(p2_object)}, Name: {p2_object.name}, Alive: {p2_object.alive}")
            else:
                print("[DEBUG_P2_INPUT] P2 Object is None")
            # ---- END DEBUG BLOCK ----

            if p2_object and p2_object.alive:
                # Get the method from p2_object based on p2_action_name_this_tick
                # Default to p2_object.Still if the action_name is somehow not a valid method (should not happen with current setup)
                action_method_to_call = getattr(p2_object, p2_action_name_this_tick, p2_object.Still)
                action_method_to_call() # Execute the action
                
                if p2_action_name_this_tick != "Still":
                    print(f"[MAIN_LOOP] Human (P2) action: {p2_action_name_this_tick}")
                # else: P2 is Still by default, no explicit print unless debugging further
            elif p2_object and not p2_object.alive:
                p2_object.Still() # Dead P2 does nothing
                # print("[MAIN_LOOP] Human (P2) is dead, action: Still")
            # else: p2_object is None (should not happen if initialized correctly)

            # Conditionally execute AI's turn and game step
            if p2_action_name_this_tick != "Still" or step_count == 0:
                print(f"--- Game Tick (Actual Step: {step_count + 1}) ---")
                # Agent's turn (P1)
                # AI Agent (Player 1) action
                ai_action, _ = model.predict(stacked_observation_ai_batch, deterministic=args.deterministic)
                ai_action = ai_action[0] # Remove batch dimension from action

                # Environment step. game_env.step() takes AI's (Player 1) action.
                new_raw_obs_p1, reward_p1, terminated, truncated, info = game_env.step(ai_action)
                
                # Update AI's observation deque with its new observation
                obs_deque.append(new_raw_obs_p1)
                
                episode_reward_player1 += reward_p1
                # Player 2's reward for the step, if provided by the environment
                if 'player2_reward_this_step' in info: 
                     episode_reward_player2 += info['player2_reward_this_step']

                print(f"AI (P1) chose action: {ai_action}, Received reward: {reward_p1:.2f}")
                if 'player2_action_taken' in info: # Action P2 took, as processed by env
                    print(f"Human (P2) confirmed action by env: {info['player2_action_taken']}")
                if 'rewards_breakdown' in info and info['rewards_breakdown']: # AI's reward breakdown
                    print(f"  AI (P1) Rewards Breakdown: {info['rewards_breakdown']}")
                
                # This check for 'human_player_quit' from info dict might be redundant
                # if Pygame 'Q' key event handling is the primary quit mechanism.
                # if info.get('human_player_quit', False):
                #     print("Human player quit the game (detected via info dict).")
                #     terminated = True # Ensure termination if info dict signals it

                step_count += 1 # Increment game step counter only when a game tick occurs
            elif p2_action_name_this_tick == "Still": # P2 is Still, and it's not the first step (step_count > 0)
                print(f"[MAIN_LOOP] Human (P2) is Still. Game tick skipped. (Visual Loop Step: {step_count + 1})")
            # If P2 made a move (p2_action_name_this_tick != "Still") but it was not step_count == 0, it's covered by the first 'if'.

            if terminated or truncated:
                game_env.render() # Render final state
                print("\n--- Game Over ---")
                p1_obj = game_env.game.player_name_to_object.get('P1')
                p2_obj = game_env.game.player_name_to_object.get('P2')

                if p1_obj and p2_obj: # Ensure both player objects were retrieved
                    if not p1_obj.alive and p2_obj.alive:
                        print("You (Player 2) win! AI (Player 1) died.")
                    elif not p2_obj.alive and p1_obj.alive:
                        print("AI (Player 1) wins! You (Player 2) died.")
                    elif not p1_obj.alive and not p2_obj.alive:
                        print("It's a draw! Both players died.")
                    elif truncated:
                        print("Game ended due to timeout (max steps reached).")
                    else:
                        # This case implies terminated is True but not due to player death or truncation.
                        # Could be a specific win condition from info['winner'] if that was implemented.
                        print("Game ended. (No specific win/loss by death or timeout)")
                elif truncated:
                    print("Game ended due to timeout (max steps reached), player status unclear.")
                else:
                    print("Game ended for an unknown reason, player status unclear.")
                break
            
            # Optional: Small delay to make the game more playable/watchable
            # time.sleep(0.1) 

    except KeyboardInterrupt:
        print("\nGame interrupted by user (Ctrl+C).")
    finally:
        print(f"\nFinal Score: AI (P1): {episode_reward_player1:.2f}, Human (P2): {episode_reward_player2:.2f}")
        game_env.close()
        print("Game environment closed.")

if __name__ == '__main__':
    main()
