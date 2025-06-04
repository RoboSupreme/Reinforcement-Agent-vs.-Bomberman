import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
from game import Game, Player # Added Player import
from collections import deque

# Ensure game.py is in the same directory or accessible in PYTHONPATH
# and that it contains the Game class with the get_predicted_blast_coords method
# and the Player class.
from game import Game, Player 

# Constants for observation layers
OBS_LAYER_BOARD = 0
OBS_LAYER_PLAYER_POS = 1
OBS_LAYER_BOMB_LOCATIONS = 2
OBS_LAYER_DANGER_TTE0 = 3 # Exploding this tick
OBS_LAYER_DANGER_TTE1 = 4 # Exploding in 1 tick
OBS_LAYER_DANGER_TTE2 = 5 # Exploding in 2 ticks
OBS_LAYER_DANGER_TTE3 = 6 # Exploding in 3 ticks
OBS_LAYER_DANGER_TTE4 = 7 # Exploding in 4 ticks (just placed)
NUM_OBS_LAYERS = 8

# Tile values for the board layer in observation
OBS_TILE_SOFT_WALL = 0.0  # game.py board value 0
OBS_TILE_FLOOR = 1.0      # game.py board value 1
OBS_TILE_HARD_WALL = 2.0  # game.py board value 2

# Action mapping
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STILL = 4
ACTION_BOMB = 5

class BomberEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, map_scheme=None, render_mode=None, max_episode_steps=1000, player_controlled=None, opponent_player_num=None, opponent_style="static"):
        super().__init__()

        if map_scheme is None:
            self.map_scheme = {"name": "standard", "size": (7, 11)} # Default map
        else:
            self.map_scheme = map_scheme
        
        self.player_controlled_name = None # Will store actual game name 'P1' or 'P2' if human plays
        self.ai_player_name = None         # Will store actual game name 'P1' or 'P2' for AI

        self.game = Game(map_scheme=self.map_scheme) # Initialize game first

        # Player.__init__ takes game_instance and player_name, and calls _add_player internally.
        player1 = Player(self.game, "P1")
        player2 = Player(self.game, "P2")

        self.game.start() # Now players are loaded, so start should pass

        # Player objects are now in self.game.player_name_to_object
        # We need to identify which game.Player object is the AI and which is human (if any)
        if player_controlled: # player_controlled is the input string like 'player2'
            # Determine AI and Human player actual game names ('P1' or 'P2')
            if opponent_player_num == 1:
                self.ai_player_name = "P1"
                self.player_controlled_name = "P2" # Actual game name for the human
            elif opponent_player_num == 2:
                self.ai_player_name = "P2"
                self.player_controlled_name = "P1" # Actual game name for the human
            else: # Default if opponent_player_num is not specified or invalid (e.g. play_vs_agent.py sets opponent_player_num=1)
                self.ai_player_name = "P1"
                self.player_controlled_name = "P2" # Actual game name for the human
            
            self.ai_player_object = self.game.player_name_to_object.get(self.ai_player_name)
            self.human_player_object = self.game.player_name_to_object.get(self.player_controlled_name)
            
            self.player = self.ai_player_object # AI is the primary agent for the env's action space
            self.opponent = self.human_player_object
            self.opponent_style = "human" # If player_controlled, the opponent is human

            if not self.ai_player_object:
                raise ValueError(f"AI player '{self.ai_player_name}' not found. Available: {list(self.game.player_name_to_object.keys())}")
            if not self.human_player_object:
                raise ValueError(f"Human player '{self.player_controlled_name}' (derived from input '{player_controlled}') not found. Available: {list(self.game.player_name_to_object.keys())}")
            print(f"[INFO] Env Init: AI is {self.ai_player_name}, Human is {self.player_controlled_name} (input ID: '{player_controlled}')")

        else: # Standard AI vs AI or AI vs Static opponent (default P1 is agent)
            self.ai_player_name = "P1"
            self.player = self.game.player_name_to_object.get("P1") # Agent is P1
            self.opponent = self.game.player_name_to_object.get("P2") # Opponent is P2 (can be static or another AI)
            self.player_controlled_name = None
            self.human_player_object = None
            self.opponent_style = opponent_style # For AI vs AI/Static, use the passed or default style
            if not self.player:
                raise ValueError(f"Player P1 not found in game.players. Available: {list(self.game.player_name_to_object.keys())}")
            if not self.opponent:
                # This case is fine if it's meant to be single player training
                print(f"[WARN] Player P2 not found. Agent will play alone or vs static if game logic implies.")
            print(f"[INFO] Env Init: Agent is {self.ai_player_name}, Opponent is P2 (or static)")

        self.action_space = spaces.Discrete(6) # 0:Up, 1:Down, 2:Left, 3:Right, 4:Still, 5:Bomb

        board_shape = self.game.board.shape
        self.observation_space = spaces.Box(
            low=0, high=max(OBS_TILE_HARD_WALL, 1.0), # Max value in any layer (binary for pos/bomb/danger)
            shape=(board_shape[0], board_shape[1], NUM_OBS_LAYERS),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        self.current_step_in_episode = 0
        # self.player is now assigned based on player_controlled logic above
        self._reset_reward_trackers() # Call after self.player is potentially available

        # Key mapping for human player (Player 2 in play_vs_agent.py)
        # These map to Player object methods: Up(), Down(), Left(), Right(), Still(), Bomb()
        self.human_key_map = {
            pygame.K_w: 'Up',
            pygame.K_s: 'Down',
            pygame.K_a: 'Left',
            pygame.K_d: 'Right',
            pygame.K_SPACE: 'Bomb',
            pygame.K_e: 'Still' # 'e' for 'stand still' or 'easy'
        }
        self.quit_key = pygame.K_q

        # Pygame rendering attributes
        self.screen = None
        self.clock = None
        self.tile_size = 50 
        self.font = None
        if self.render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Bomberman RL Environment (New)")
        board_height, board_width = self.game.board.shape
        self.screen_width = board_width * self.tile_size
        self.screen_height = board_height * self.tile_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        try:
            # Try a common system font with a larger size
            self.font = pygame.font.SysFont('arial', 30) # Or 'comicsansms', 'tahoma', etc.
        except: # Fallback font if system font isn't found
            self.font = pygame.font.Font(None, 36) # Default pygame font, make it a bit larger
        self._load_colors()

        # Load player and opponent sprites
        self.player_image = None
        self.opponent_image = None
        asset_dir = os.path.join(os.path.dirname(__file__), "Asset")
        player_sprite_path = os.path.join(asset_dir, "player1.png")
        opponent_sprite_path = os.path.join(asset_dir, "player2.png")

        new_sprite_dim = int(self.tile_size * 2)
        try:
            player_img_raw = pygame.image.load(player_sprite_path).convert_alpha()
            self.player_image = pygame.transform.scale(player_img_raw, (new_sprite_dim, new_sprite_dim))
            print(f"Successfully loaded player sprite from {player_sprite_path} and scaled to {new_sprite_dim}x{new_sprite_dim}")
        except pygame.error as e:
            print(f"Error loading player sprite from {player_sprite_path}: {e}. Will use text rendering.")

        try:
            opponent_img_raw = pygame.image.load(opponent_sprite_path).convert_alpha()
            self.opponent_image = pygame.transform.scale(opponent_img_raw, (new_sprite_dim, new_sprite_dim))
            print(f"Successfully loaded opponent sprite from {opponent_sprite_path} and scaled to {new_sprite_dim}x{new_sprite_dim}")
        except pygame.error as e:
            print(f"Error loading opponent sprite from {opponent_sprite_path}: {e}. Will use text rendering.")

    def _load_colors(self):
        self.colors = {
            "soft_wall": (139, 69, 19),     # SaddleBrown (Darker)
            "floor": (230, 230, 230),       # Very Light Grey (Brighter)
            "hard_wall": (60, 60, 60),      # Dark Grey (Darker)
            "player": (0, 255, 255),        # Bright Cyan (unchanged)
            "opponent": (50, 205, 50),      # Lime Green (unchanged)
            "bomb": (255, 0, 0),            # Bright Red (unchanged)
            "explosion": (255, 100, 0),     # Fiery Orange (unchanged)
            "grid_line": (100, 100, 100),   # Lighter Grey (Adjusted for new hard_wall)
            "danger_zone_colors": [ # Colors for TTE 0 to 4 (unchanged)
                pygame.Color(255, 0, 0, 180),    # TTE 0 (Intense Red)
                pygame.Color(255, 100, 0, 150),  # TTE 1 (Bright Orange)
                pygame.Color(255, 255, 0, 120),  # TTE 2 (Yellow)
                pygame.Color(255, 105, 180, 90), # TTE 3 (Hot Pink)
                pygame.Color(0, 191, 255, 70)    # TTE 4 (Deep Sky Blue)
            ]
        }

    def _reset_reward_trackers(self):
        self.action_history = deque(maxlen=3)
        self.visited_tiles = set()
        # Ensure player and player.position exist before adding to visited_tiles
        if hasattr(self, 'player') and self.player and self.player.position:
             self.visited_tiles.add(self.player.position)
        self.turns_since_last_wall_break = 0
        # Ensure game object and get_wall_stats method exist
        if hasattr(self, 'game') and self.game:
            self.total_walls_at_episode_start = self.game.get_wall_stats()["current_walls"]
        else: # Should not happen if __init__ is correct
            self.total_walls_at_episode_start = 0 

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Layer 0: Board state
        board_obs = np.full(self.game.board.shape, OBS_TILE_FLOOR, dtype=np.float32)
        board_obs[self.game.board == 0] = OBS_TILE_SOFT_WALL
        board_obs[self.game.board == 2] = OBS_TILE_HARD_WALL
        obs[:, :, OBS_LAYER_BOARD] = board_obs

        # Layer 1: Player position
        if self.player.alive and self.player.position:
            obs[self.player.position[0], self.player.position[1], OBS_LAYER_PLAYER_POS] = 1.0

        # Layer 2: Bomb locations
        # Layer 3-7: Danger zones (TTE 0-4)
        # game._bomb_dict stores timer values from 1 (just placed) to 5 (exploding now)
        for r_idx in range(self.game.board.shape[0]):
            for c_idx in range(self.game.board.shape[1]):
                bomb_timer_value = self.game._bomb_dict[r_idx][c_idx]
                if bomb_timer_value > 0: # There's a bomb or its precursor
                    # Mark bomb location
                    obs[r_idx, c_idx, OBS_LAYER_BOMB_LOCATIONS] = 1.0
                    
                    # Determine TTE and corresponding danger layer
                    # Timer 5 means exploding now (TTE=0)
                    # Timer 1 means just placed (TTE=4)
                    tte = 5 - bomb_timer_value 
                    if 0 <= tte <= 4:
                        danger_layer_index = OBS_LAYER_DANGER_TTE0 + tte
                        
                        current_bomb_obj = None
                        for bomb_obj_in_list in self.game.bombs:
                            if bomb_obj_in_list.position == (r_idx, c_idx) and not bomb_obj_in_list.exploded:
                                current_bomb_obj = bomb_obj_in_list
                                break
                        
                        if current_bomb_obj:
                            blast_radius = current_bomb_obj.blast_radius
                            predicted_coords = self.game.get_predicted_blast_coords(r_idx, c_idx, blast_radius)
                            for br, bc in predicted_coords:
                                if 0 <= br < obs.shape[0] and 0 <= bc < obs.shape[1]: # Boundary check for safety
                                    obs[br, bc, danger_layer_index] = 1.0 # Mark danger
        return obs

    def _opponent_move(self, opponent_player_object):
        """
        Determines and queues an action for a non-human opponent based on its style.
        """
        if not opponent_player_object or not opponent_player_object.alive:
            return # No action if opponent is None or not alive

        if self.opponent_style == "static":
            opponent_player_object.Still()
        # TODO: Implement other opponent styles like "random", "aggressive", etc.
        # elif self.opponent_style == "random":
        #     # Example: choose a random action
        #     possible_actions = [
        #         opponent_player_object.Up,
        #         opponent_player_object.Down,
        #         opponent_player_object.Left,
        #         opponent_player_object.Right,
        #         opponent_player_object.Still,
        #         opponent_player_object.Bomb
        #     ]
        #     import random
        #     action_method = random.choice(possible_actions)
        #     action_method()
        else:
            # Default to Still if style is unknown or not implemented
            # print(f"[WARN] Unknown opponent_style '{self.opponent_style}'. Opponent will remain still.") # Uncomment for debugging
            opponent_player_object.Still()

    def step(self, action):
        self.current_step_in_episode += 1
        reward = 0.0
        info = {
            "rewards_breakdown": {}, 
            "player_pos": None, 
            "player_alive": None, 
            "action_taken_successfully": False, 
            "bombs_active": 0, 
            "current_walls": 0, 
            "walls_broken": 0,
            "player2_action_taken": "Still", # Default for human player, updated if human plays
            "human_player_quit": False,
            "player2_reward_this_step": 0.0 # Placeholder for human player's reward if calculated
        }

        # Store state before action for reward calculation
        walls_before_action = self.game.get_wall_stats()["current_walls"]
        board_state_before_action = self.game.board.copy()
        player_pos_before_action = self.player.position

        # 1. Perform action / Set player action in queue
        action_taken_successfully = False
        if self.player.alive:
            if action == ACTION_UP: action_taken_successfully = self.player.Up()
            elif action == ACTION_DOWN: action_taken_successfully = self.player.Down()
            elif action == ACTION_LEFT: action_taken_successfully = self.player.Left()
            elif action == ACTION_RIGHT: action_taken_successfully = self.player.Right()
            elif action == ACTION_STILL: action_taken_successfully = self.player.Still()
            elif action == ACTION_BOMB:
                action_taken_successfully = self.player.Bomb()
                if action_taken_successfully: # Bomb was placed
                    bomb_r, bomb_c = self.player.position
                    if self.game._bomb_dict[bomb_r][bomb_c] > 0:
                        blast_radius = self.game.game_rules.get("bomb_blast_radius", 2)
                        predicted_blast = self.game.get_predicted_blast_coords(bomb_r, bomb_c, blast_radius)
                        hits_wall = False
                        for pr, pc in predicted_blast:
                            if 0 <= pr < board_state_before_action.shape[0] and 0 <= pc < board_state_before_action.shape[1]:
                                if board_state_before_action[pr, pc] == 0 or board_state_before_action[pr, pc] == 2:
                                    hits_wall = True
                                    break
                        if hits_wall:
                            reward += 5.0
                            info["rewards_breakdown"]["strategic_bomb_placement"] = 5.0
            else:
                # Fallback for unexpected action value from agent
                # print(f"[WARN] BomberEnv: Received unexpected action {action} from agent. Defaulting to STILL.") # Uncomment for debugging
                self.player.Still() # Ensure an action is queued for the agent
                action_taken_successfully = True # Or False, depending on definition for forced Still
        else:
            # Agent is dead, queue a 'Still' action
            self.player.Still()
            action_taken_successfully = True

        # 2. Handle Opponent (P2) action if it's an AI.
        #    If P2 is human-controlled, their actions are performed EXTERNALLY in play_vs_agent.py
        #    directly on the P2 player object before this env.step() is called.
        #    The self.opponent object (initialised as P2) will have its action queue populated by then.
        #    If P2 is an AI opponent, its action needs to be determined here.

        _player2_action_name_this_tick = "Still (External/Pre-set)" # Default for info dict

        if self.opponent and self.opponent.name == self.player_controlled_name: # This is P2, the human player
            # Human player's action was already set externally. Nothing to do here for P2's action decision.
            # We can retrieve the action P2 is about to take if game.py stores it before processing, or assume Still for info.
            # For now, we assume the action is queued in the P2 player object by the external script.
            # If self.human_player_object.last_action_name is a thing, we could use it.
            # For simplicity, we'll just report a generic status in info.
            pass # P2 human action is handled externally
        
        elif self.opponent and self.opponent.alive: # This is P2, and it's an AI opponent
            # This case is for when P1 is AI, and P2 is also an AI (not human)
            # or P1 is Human, and P2 is an AI.
            # The 'opponent_style' attribute would typically determine the AI's behavior.
            # For now, let's assume a simple AI opponent logic if one were to be implemented here.
            # This relies on self.opponent being correctly set to the P2 object and not being human-controlled.
            # The existing self._opponent_move(self.opponent) can be used if self.opponent is P2 AI.
            if self.opponent_style != "human": # Ensure it's not trying to AI-control a human-designated player
                # print(f"[ENV_STEP] Processing AI opponent move for {self.opponent.name}")
                self._opponent_move(self.opponent) # self.opponent here is P2 (AI)
                # Try to get the action name if _opponent_move sets it (hypothetical)
                # if hasattr(self.opponent, 'last_action_name_queued'):
                # _player2_action_name_this_tick = self.opponent.last_action_name_queued
                # else:
                _player2_action_name_this_tick = "AI Action (via _opponent_move)"
            else:
                # This case should ideally not be hit if P2 is human (opponent_style == 'human')
                # If P2 is human, their action is external. If P2 is AI, opponent_style shouldn't be 'human'.
                # Defaulting to Still for safety if logic is convoluted.
                self.opponent.Still()
                _player2_action_name_this_tick = "Still (Fallback for P2 AI)"

        elif self.opponent and not self.opponent.alive: # P2 (AI or Human) is dead
            self.opponent.Still() # Dead players do nothing
            _player2_action_name_this_tick = "Still (Dead)"
        
        # else: P2 (self.opponent) is None - single player mode, no P2 action to consider.

        info["player2_action_taken"] = _player2_action_name_this_tick

        # 3. Update game state (processes all queued actions for P1 and P2, bomb timers, explosions)
        self.game.update_frame()

        player_pos_after_action = self.player.position # Get new position after game update
        # Penalty for ineffective movement (actions 0-3: Up, Down, Left, Right)
        # Applied if the agent chose a move action but its position didn't change.
        if action in [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]:
            # player_pos_before_action is captured at the start of the step.
            # If action is a move action, it implies self.player.alive was true at that time.
            if player_pos_before_action is not None and player_pos_after_action is not None and \
               player_pos_before_action == player_pos_after_action: # Direct tuple comparison
                ineffective_move_penalty = -5.0
                reward += ineffective_move_penalty
                info["rewards_breakdown"]["ineffective_move_penalty"] = ineffective_move_penalty

        # 3. Calculate rewards based on outcome (Note: comment numbering ideally should be adjusted)
        walls_after_action = self.game.get_wall_stats()["current_walls"]
        walls_broken_this_step = walls_before_action - walls_after_action

        if walls_broken_this_step > 0:
            wall_break_reward = 50.0 * walls_broken_this_step
            reward += wall_break_reward
            info["rewards_breakdown"]["wall_broken"] = wall_break_reward
            self.turns_since_last_wall_break = 0
            if self.player.alive: # Survived the step where walls broke
                survival_bonus = 100.0
                reward += survival_bonus
                info["rewards_breakdown"]["wall_break_survival_bonus"] = survival_bonus
        else:
            if self.game.get_wall_stats()["current_walls"] > 0: # Only increment if walls still exist
                 self.turns_since_last_wall_break += 1

        # Stagnation penalty
        if self.turns_since_last_wall_break >= 10 and self.game.get_wall_stats()["current_walls"] > 0:
            stagnation_penalty = -20.0
            reward += stagnation_penalty
            info["rewards_breakdown"]["stagnation_penalty"] = stagnation_penalty
            self.turns_since_last_wall_break = 0 # Reset counter after penalty

        # Exploration reward
        if self.player.alive and self.player.position not in self.visited_tiles:
            self.visited_tiles.add(self.player.position)
            exploration_reward = 100.0
            reward += exploration_reward
            info["rewards_breakdown"]["exploration"] = exploration_reward

        # Repetitive action penalty
        self.action_history.append(action)
        if len(self.action_history) == 3:
            if self.action_history[0] == self.action_history[1] == self.action_history[2]:
                if action == ACTION_STILL:
                    repetitive_penalty = -5.0
                    reward += repetitive_penalty
                    info["rewards_breakdown"]["repetitive_action_still"] = repetitive_penalty
                    self.action_history.clear()
                elif action == ACTION_BOMB:
                    repetitive_penalty = -5.0
                    reward += repetitive_penalty
                    info["rewards_breakdown"]["repetitive_action_bomb"] = repetitive_penalty
                    self.action_history.clear()
        
        # Successful Dodge Reward
        if self.player.alive and player_pos_before_action is not None:
            r_before, c_before = player_pos_before_action
            # Check if the tile the player was on at the start of the turn has exploded
            # and the player moved to a new, safe tile.
            # BLAST_TILE is assumed to be 4.
            if 0 <= r_before < self.game.board.shape[0] and 0 <= c_before < self.game.board.shape[1] and \
               self.game.board[r_before, c_before] == 4: # Tile where player was is now an explosion
                
                current_player_pos = self.player.position
                if current_player_pos != player_pos_before_action: # Player moved
                    r_current, c_current = current_player_pos
                    if 0 <= r_current < self.game.board.shape[0] and 0 <= c_current < self.game.board.shape[1] and \
                       self.game.board[r_current, c_current] != 4: # Player is not in an explosion now
                        dodge_reward = 50.0 
                        reward += dodge_reward
                        info["rewards_breakdown"]["successful_dodge"] = dodge_reward
        
        # 4. Get new observation
        obs = self._get_obs()

        # 5. Determine termination and truncation
        terminated = not self.player.alive or self.game.get_wall_stats()["current_walls"] == 0
        truncated = self.current_step_in_episode >= self.max_episode_steps
        
        if terminated and not self.player.alive:
            death_penalty = -200.0
            reward += death_penalty
            info["rewards_breakdown"]["death_penalty"] = death_penalty
            info["player_death_reason"] = "Agent died" # game.py doesn't provide specific reasons directly

        # Ensure all final info keys are set directly before returning
        info["player_pos"] = self.player.position
        info["player_alive"] = self.player.alive
        info["action_taken_successfully"] = action_taken_successfully
        info["bombs_active"] = len(self.game.bombs)
        info["current_walls"] = walls_after_action # Required by Monitor
        info["walls_broken"] = walls_broken_this_step # Required by Monitor (walls broken in this step)
        # info["rewards_breakdown"] is populated throughout the step method

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Re-instantiate the Game object
        self.game = Game(map_scheme=self.map_scheme)

        # Player.__init__ takes game_instance and player_name, and calls _add_player internally.
        # These variables player1 and player2 might be flagged as unused, but their creation is crucial
        # as it adds them to the game instance.
        player1 = Player(self.game, "P1") # noqa: F841, pylint: disable=unused-variable
        player2 = Player(self.game, "P2") # noqa: F841, pylint: disable=unused-variable

        self.game.start() # Now players are loaded, so start should pass

        # Re-link player objects using names stored during __init__
        if self.player_controlled_name: # Human vs AI mode
            self.ai_player_object = self.game.player_name_to_object.get(self.ai_player_name)
            self.human_player_object = self.game.player_name_to_object.get(self.player_controlled_name)
            
            self.player = self.ai_player_object  # self.player is the AI agent
            self.opponent = self.human_player_object # self.opponent is the human player

            if not self.ai_player_object:
                raise RuntimeError(f"AI player '{self.ai_player_name}' could not be re-linked after game reset. Available: {list(self.game.player_name_to_object.keys())}")
            if not self.human_player_object:
                raise RuntimeError(f"Human player '{self.player_controlled_name}' could not be re-linked after game reset. Available: {list(self.game.player_name_to_object.keys())}")
            # print(f"[DEBUG] Env Reset: AI is {self.ai_player_name} ({type(self.ai_player_object)}), Human is {self.player_controlled_name} ({type(self.human_player_object)})")

        else: # Standard AI vs AI or AI vs Static opponent mode
            # self.ai_player_name was set in __init__ (e.g., "P1")
            self.ai_player_object = self.game.player_name_to_object.get(self.ai_player_name)
            self.player = self.ai_player_object # self.player is the AI agent
            
            # Default opponent name is "P2" if not human controlled
            self.opponent = self.game.player_name_to_object.get("P2") 
            self.human_player_object = None # Ensure this is None in this mode

            if not self.player:
                raise RuntimeError(f"AI player '{self.ai_player_name}' could not be re-linked after game reset. Available: {list(self.game.player_name_to_object.keys())}")
            if not self.opponent:
                 print(f"[WARN] Env Reset: Opponent P2 not found after reset. Agent may play alone or vs static elements.")
            # print(f"[DEBUG] Env Reset: Agent is {self.ai_player_name} ({type(self.player)}), Opponent is P2 ({type(self.opponent)})")

        self.current_step_in_episode = 0
        self._reset_reward_trackers() # This clears self.visited_tiles among other things
        
        # Add initial positions to visited_tiles after game.start() and _reset_reward_trackers()
        # Ensure self.player (AI) is valid before accessing attributes
        if self.player and hasattr(self.player, 'alive') and self.player.alive and hasattr(self.player, 'position') and self.player.position:
            self.visited_tiles.add(self.player.position)
        
        # Ensure self.human_player_object is valid before accessing attributes
        if self.human_player_object and hasattr(self.human_player_object, 'alive') and self.human_player_object.alive and \
           hasattr(self.human_player_object, 'position') and self.human_player_object.position and \
           self.human_player_object != self.player: # Avoid double-adding if AI is also human (should not happen)
            self.visited_tiles.add(self.human_player_object.position)
        
        initial_obs = self._get_obs() # Observation is for self.player (the AI)
        info = {
            "player2_action_taken": None, 
            "human_player_quit": False,
            # Add other relevant initial info if needed by play_vs_agent.py or wrappers
        } 
        return initial_obs, info

    def render(self):
        if self.render_mode is None:
            return

        # ... (rest of the code remains the same)
            self._init_pygame()

        board = self.game.board
        player_pos = self.player.position if self.player.alive else None
        opponent_pos = self.opponent.position if self.opponent.alive else None
        
        game_surface = pygame.Surface((self.game.board.shape[1] * self.tile_size, self.game.board.shape[0] * self.tile_size))
        game_surface.fill(self.colors["floor"])

        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                if board[r, c] == 0: pygame.draw.rect(game_surface, self.colors["soft_wall"], rect)
                elif board[r, c] == 1: pygame.draw.rect(game_surface, self.colors["floor"], rect)
                elif board[r, c] == 2: pygame.draw.rect(game_surface, self.colors["hard_wall"], rect)
                pygame.draw.rect(game_surface, self.colors["grid_line"], rect, 1)

        current_obs_for_render = self._get_obs() 
        for tte_idx in range(5):
            layer_idx = OBS_LAYER_DANGER_TTE0 + tte_idx
            danger_color = self.colors["danger_zone_colors"][tte_idx]
            for r in range(board.shape[0]):
                for c in range(board.shape[1]):
                    if current_obs_for_render[r, c, layer_idx] > 0:
                        s = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                        s.fill(danger_color)
                        game_surface.blit(s, (c * self.tile_size, r * self.tile_size))

        for bomb_obj in self.game.bombs:
            if not bomb_obj.exploded:
                r, c = bomb_obj.position
                rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.ellipse(game_surface, self.colors["bomb"], rect.inflate(-self.tile_size*0.2, -self.tile_size*0.2))
                timer_val = self.game._bomb_dict[r][c]
                tte = 5 - timer_val
                if self.font and 0 <= tte <= 4:
                    text_surf = self.font.render(str(tte), True, (255,255,255))
                    text_rect = text_surf.get_rect(center=rect.center)
                    game_surface.blit(text_surf, text_rect)

        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if 6 in self.game.movable_objects[r][c]:
                    rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                    pygame.draw.rect(game_surface, self.colors["explosion"], rect)

        if player_pos:
            tile_rect = pygame.Rect(player_pos[1] * self.tile_size, player_pos[0] * self.tile_size, self.tile_size, self.tile_size)
            if self.player_image:
                # Center the larger sprite image on the tile
                player_image_rect = self.player_image.get_rect(center=tile_rect.center)
                game_surface.blit(self.player_image, player_image_rect.topleft)
            elif self.font: # Fallback to text if image not loaded
                player_char_surf = self.font.render("P", True, self.colors["player"])
                player_char_rect = player_char_surf.get_rect(center=tile_rect.center)
                game_surface.blit(player_char_surf, player_char_rect)
    
        if opponent_pos:
            tile_rect = pygame.Rect(opponent_pos[1] * self.tile_size, opponent_pos[0] * self.tile_size, self.tile_size, self.tile_size)
            if self.opponent_image:
                # Center the larger sprite image on the tile
                opponent_image_rect = self.opponent_image.get_rect(center=tile_rect.center)
                game_surface.blit(self.opponent_image, opponent_image_rect.topleft)
            elif self.font: # Fallback to text if image not loaded
                opponent_char_surf = self.font.render("O", True, self.colors["opponent"])
                opponent_char_rect = opponent_char_surf.get_rect(center=tile_rect.center)
                game_surface.blit(opponent_char_surf, opponent_char_rect)

        if self.render_mode == "human":
            self.screen.blit(game_surface, (0,0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(game_surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

# Example usage (optional, for testing)
if __name__ == '__main__':
    env = BomberEnv(render_mode='human')
    obs, info = env.reset()
    done = False
    total_reward_acc = 0
    for _ in range(500):
        action = env.action_space.sample() # Sample random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_acc += reward
        env.render()
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward_acc}")
            print(f"Termination reason: Player alive? {env.player.alive}, Walls left: {env.game.get_wall_stats()['current_walls']}")
            obs, info = env.reset()
            total_reward_acc = 0
        # Allow window to close by checking for pygame.QUIT event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True # Set done to true to break the loop
                break
        if done: # Break outer loop if window was closed
            break
    env.close()
