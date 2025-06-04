import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game, Player
from render_tool import RenderTool, MapScheme
from collections import deque
from pathlib import Path

BOARD_H, BOARD_W = 7, 11
MAX_BOMBS        = 3           # for action mask

class BomberEnv(gym.Env):
    METADATA = {"render_modes": ["human", "rgb_array"], "render_fps": 8}
    BOARD_H, BOARD_W = 7, 11 # Assuming these are defined or accessible

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode=None, opponent="still", curriculum_stage=None, verbose=0):
        super().__init__()
        self.verbose = verbose # For logging curriculum changes
        self.map_scheme     = MapScheme().standard
        self.render_mode    = render_mode
        self.opponent_style = opponent
        self.curriculum_stage_name = curriculum_stage # Store curriculum stage name
        self._apply_curriculum_settings() # Apply initial settings
        
        # DEBUG flags to track agent performance
        self.debug_episode_deaths = 0
        self.debug_death_causes = {"self_bomb": 0, "enemy_bomb": 0, "timeout": 0}
        self.debug_avg_survival_time = 0
        self.debug_total_episodes = 0
        self.debug_walls_broken_total = 0
        # 6 discrete actions
        self.action_space = spaces.Discrete(6)
        # 7-plane binary board with enhanced information
        # 0: void, 1: land, 2: block, 3: bombs, 4: players, 5: danger zones, 6: potentially accessible areas
        self.observation_space = spaces.Box(0, 1,
            shape=(BOARD_H, BOARD_W, 7), dtype=np.float32)
        
        self.MAX_EPISODE_STEPS = 5000 # Maximum steps per episode before timeout

        # For tracking idle behavior and position history
        self.pos_history = deque(maxlen=4)  # Need to track at least 3 positions + current for the 3-turn idle check
        self.idle_counter = 0  # Will be used differently - now tracks consecutive turns without moving
        self.prev_in_blast = False
        self.current_blast = set()
        
        # For tracking bomb placement and blocks
        self.prev_bomb_count = 0
        self.current_blocks = set()
        self.last_action_was_bomb = False
        self.prev_bomb_positions = set()  # Track bomb positions to discourage spam
        
        # For encouraging position diversity
        self.recent_positions = deque(maxlen=10)  # Track recent positions to reward diversity

        # For penalizing repetitive actions
        self.action_history_len = 3
        self.action_history = deque(maxlen=self.action_history_len)
        self.penalty_repetitive_action = -25 # Penalty for repeating the same action 3 times

        # For new reward system
        self.board_state_before_action = None
        self.agent_bombed_this_turn_coords = None # (r,c) if agent bombed, else None
        self.was_in_blast_at_start_of_turn = False
        # self.current_voids = set() # This will be calculated dynamically or from board_state_before_action
        

        # Attributes for revamped reward system (inspired by old system)
        self.p1_last_action_was_bomb = False
        self.p1_bomb_spam_locations = set() # Tracks where P1 has bombed in current episode
        self.p1_active_bombs_count_before_action = 0
        self.walls_count_before_action = 0
        self.void_count_before_action = 0
        self.total_blocks_broken_episode = 0 # Track total blocks broken in an episode
        self.p1_recent_bomb_locations = set() # For old reward: tracks P1 bomb spots, cleared on P1 bomb explosion
        self.destroyed_this_episode_wall_coords = set() # For first_time_wall_break_bonus

        # --- Exploration Reward ---
        self.visited_tiles_in_episode = set()
        self.exploration_reward_value = 0.1  # Small reward for visiting a new tile

        # --- Penalty/Bonus for Wall Breaking Record ---
        self.max_walls_broken_overall = 0  # Max walls broken in any single episode so far
        self.non_max_wall_break_penalty_value = 0.0 # Penalty if episode_broken_walls <= max_walls_broken_overall (set to 0 to remove)
        self.new_max_walls_bonus_value = 100.0 # Bonus for setting a new max wall break record

        # --- Reward for Bombing New Spots ---
        self.bombed_locations_in_episode = set()

    def _apply_curriculum_settings(self):
        """Applies settings like opponent style and bomb availability based on curriculum stage."""
        self.disable_bombs = False # Set by curriculum

        # self.opponent_style is already set in __init__ or can be updated here

        if self.curriculum_stage_name == 'movement_only':
            self.opponent_style = 'still'
            self.disable_bombs = True
        elif self.curriculum_stage_name == 'still_bombs':
            self.opponent_style = 'still'
            self.disable_bombs = False
        elif self.curriculum_stage_name == 'random_bombs':
            self.opponent_style = 'random' # Assuming 'random' opponent places bombs
            self.disable_bombs = False
        elif self.curriculum_stage_name == 'full_game':
            self.opponent_style = 'random' # Or a more advanced default
            self.disable_bombs = False
        else: # Default or unknown stage
            self.opponent_style = getattr(self, 'opponent_style', 'still') # Keep existing or default to still
            self.disable_bombs = False
        
        # This attribute is directly used by get_action_mask
        # self.disable_bombs_for_curriculum = self.disable_bombs 
        # No, get_action_mask directly checks self.disable_bombs, so the above line is not needed.

    def set_curriculum_stage(self, stage_name: str):
        """Allows external callbacks to set the curriculum stage."""
        if self.verbose > 0:
            print(f"BomberEnv: Setting curriculum stage to: {stage_name}")
        self.curriculum_stage_name = stage_name
        self._apply_curriculum_settings()
        # Note: The environment will typically be reset by the callback after changing the stage,
        # which will also call _apply_curriculum_settings via reset().

    # ---------- core API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Track episode completion and update debugging stats
        if hasattr(self, 'game') and hasattr(self, 'debug_total_episodes'):
            self.debug_total_episodes += 1
            
            # Track walls broken in last episode
            if hasattr(self.game, 'get_wall_stats'):
                wall_stats = self.game.get_wall_stats()
                if 'walls_broken' in wall_stats:
                    self.debug_walls_broken_total += wall_stats['walls_broken']
        
        # Initialize game and players using the correct pattern
        self.action_history.clear()
        self.game = Game(self.map_scheme)
        self.p1 = Player(self.game, "P1")
        
        # Always create p2 regardless of opponent_style
        # The game requires both player slots to be filled
        self.p2 = Player(self.game, "P2")
            
        # Start the game
        self.game.start()

        self.viewer = RenderTool(self.game)
        self.recent_positions.clear()
        self.idle_counter = 0
        self.pos_history.clear()
        self.visited_tiles_in_episode.clear() # Reset for new episode exploration
        self.bombed_locations_in_episode.clear() # Reset for new bomb spot reward
        
        # Track current episode steps for survival time calculation
        self.current_episode_steps = 0
        
        # Reset bomb tracking for reward system
        self.p1_recent_bomb_locations.clear()
        self.p1_pending_bomb_escapes = []
        self.action_history.clear()
        self.pos_history = deque([self.p1.position], maxlen=2)
        if self.p1.position: # Ensure position is not None before adding
            self.visited_tiles_in_episode.add(self.p1.position) # Add initial position of new episode
        self.recent_positions = deque(maxlen=20)
        self.recent_positions.append(self.p1.position)  # Add initial position
        self.idle_counter = 0
        
        self.prev_bomb_count = 0
        # Store the tuple (rows, cols) for block tracking
        self.current_blocks = self._blocks()
        self.current_voids = self._void_tiles()
        self.prev_bomb_positions = set()
        self.prev_in_blast = False
        self.last_action_was_bomb = False
        self._cache_blast()  # Initialize blast tiles cache

        # Initialize for new reward system
        if self.game and hasattr(self.game, 'board'): # Ensure game is initialized
            self.board_state_before_action = self.game.board.copy()
            self.current_voids = self._void_tiles()
        else: # Should not happen if reset is called after __init__ fully
            self.board_state_before_action = np.zeros((BOARD_H, BOARD_W))
            self.current_voids = (np.array([]), np.array([]))
        self.agent_bombed_this_turn_coords = None
        self.was_in_blast_at_start_of_turn = False

    # Initialize for revamped reward system
        self.p1_last_action_was_bomb = False
        self.p1_bomb_spam_locations.clear()
        self.p1_recent_bomb_locations.clear() # For old reward logic
        if self.game and self.p1: # Ensure game and p1 are initialized
            self.p1_active_bombs_count_before_action = self._get_player_active_bombs_count(self.p1)
        else:
            self.p1_active_bombs_count_before_action = 0
    
        if self.game and hasattr(self.game, 'board'):
            self.walls_count_before_action = np.sum(self.game.board == 2) # Wall tile type is 2
            self.void_count_before_action = np.sum(self.game.board == 0)  # Void tile type is 0
        else:
            self.walls_count_before_action = 0
            self.void_count_before_action = 0
        self.total_blocks_broken_episode = 0 # Reset for the new episode

        # Apply curriculum settings based on self.curriculum_stage_name
        self._apply_curriculum_settings()
        
        # Reset exploration tracking for the new episode
        self.visited_tiles_in_episode.clear()
        # Initial position will be added after p1 is initialized and game started
        
        # Get the initial observation to return
        obs = self._obs()
        
        # Return observation and empty info dict (as per Gymnasium API)
        return obs, {}

    def step(self, action:int):
        # Track episode progress
        self.current_episode_steps += 1
        
        # Store pre-action position for comparison later
        p1_pre_action_pos = None
        if hasattr(self, 'p1') and self.p1 and hasattr(self.p1, 'position'):
            p1_pre_action_pos = self.p1.position
            # Store current position in history for urgency-based rewards
            self.pos_history.append(self.p1.position)
            
        self.action_history.append(action)
        # For new reward system: store state *before* agent's action and game update
        self.board_state_before_action = self.game.board.copy() # General board state
        self.walls_count_before_action = np.sum(self.game.board == 2)
        self.void_count_before_action = np.sum(self.game.board == 0)
        self.was_in_blast_at_start_of_turn = self._in_blast(self.p1.position)
        self.p1_active_bombs_count_before_action = self._get_player_active_bombs_count(self.p1)
        self.p1_last_action_was_bomb = (action == 4) # Action 4 is Bomb
        strategic_bomb_placement_reward = 0.0 # Initialize reward for this step
        
        # Track if agent places a bomb this turn
        # Action 4 is Bomb. Record agent's position if they bomb.
        if self.p1_last_action_was_bomb:
            self.agent_bombed_this_turn_coords = self.p1.position
            # Add this bomb to pending_bomb_escapes for more targeted escape reward
            # Based on memory: bombs explode after 4 ticks
            # Determine if the bomb placement is strategic
            bomb_pos_for_check = self.p1.position
            potential_blast_tiles = self._blast_tiles(bomb_pos_for_check[0], bomb_pos_for_check[1])
            is_strategic = False
            for r_tile, c_tile in potential_blast_tiles:
                if 0 <= r_tile < self.BOARD_H and 0 <= c_tile < self.BOARD_W:
                    tile_type_in_blast = self.board_state_before_action[r_tile, c_tile]
                    if tile_type_in_blast == 2 or tile_type_in_blast == 0: # Destructible Wall (2) or Void (0)
                        is_strategic = True
                        break
            self.p1_pending_bomb_escapes.append((self.p1.position, self.game.frame + 4, is_strategic))

            # The strategic_bomb_placement_reward is handled below based on the same 'is_strategic' logic
            # This reward is added to info['rewards'] directly in step()

            # Check for strategic bomb placement
            bomb_pos = self.agent_bombed_this_turn_coords
            # Use self.board_state_before_action to check tiles, as _blast_tiles uses current self.game.board
            # but for strategic check, we need to know what *was* there.
            # However, _blast_tiles itself uses self.game.board which at this point *is* the pre-action state for the bomb entity.
            potential_blast_tiles = self._blast_tiles(bomb_pos[0], bomb_pos[1])
            is_strategic = False
            for r_tile, c_tile in potential_blast_tiles:
                if 0 <= r_tile < self.BOARD_H and 0 <= c_tile < self.BOARD_W:
                    tile_type_in_blast = self.board_state_before_action[r_tile, c_tile]
                    if tile_type_in_blast == 2 or tile_type_in_blast == 0: # Destructible Wall (2) or Void (0)
                        is_strategic = True
                        break
            if is_strategic:
                strategic_bomb_placement_reward = 1.0 # Reward for strategic placement
            else:
                strategic_bomb_placement_reward = -1.0 # Penalty for non-strategic placement
        else:
            self.agent_bombed_this_turn_coords = None
        
        # SAFETY CHECKS REMOVED - Allow bomb placement in all situations
        # Get the action mask (but we won't use it to restrict bomb placement)
        mask = self.get_action_mask()
        
        # Execute agent's action
        self._exec(self.p1, action)
        
        # Execute opponent's action
        self._opponent_move()                   # rule-based foe
        self.game.update_frame() # This updates game state, including bomb timers, explosions
        
        # IMPORTANT: Cache blast *after* game.update_frame() so it reflects the current state for reward calculation
        self._cache_blast()

        # Update total_blocks_broken_episode
        walls_count_after_action = np.sum(self.game.board == 2) # Wall tile type is 2
        walls_broken_this_step = self.walls_count_before_action - walls_count_after_action
        if walls_broken_this_step > 0:
            self.total_blocks_broken_episode += walls_broken_this_step
        
        # Get post-action position for comparison
        p1_post_action_pos = self.p1.position if self.p1 and hasattr(self.p1, 'position') else None
        
        # Safety checks with balanced penalties and rewards
        entered_imminent_danger = False
        successful_avoidance = False
        danger_penalties = {}
        
        # Current position
        r, c = self.p1.position
        
        # 1. Check if agent walked into immediate danger (explosion time = 1)
        # This is represented by marker 5 (explodes next tick) in the game's movable objects
        if 5 in self.game.movable_objects[r][c]:  # Marker 5 = explosion next tick
            entered_imminent_danger = True
            danger_penalties['entered_imminent_danger'] = -5.0  # More moderate penalty
        
        # 2. Check if agent successfully avoided danger
        # This is a POSITIVE reinforcement for good safety behavior
        if self.current_episode_steps > 1 and self.was_in_blast_at_start_of_turn and not self._in_blast(self.p1.position):
            # Agent moved from danger to safety!
            successful_avoidance = True
            danger_penalties['successful_avoidance'] = 25.0  # Massively increased reward for avoiding danger zones
        
        # 3. Check for extreme danger situations (simplify the complex escape cost logic)
        in_danger = self._in_blast(self.p1.position)
        if in_danger:
            # Check if this is an imminent explosion (5 = explodes next tick)
            ids = self.game.movable_objects[r][c]
            if 5 in ids:
                # Only penalize being in an imminent explosion zone
                danger_penalties['imminent_explosion_zone'] = -5.0
                
                # Extra penalty if they stayed still in a tile about to explode
                if p1_pre_action_pos is not None and p1_post_action_pos == p1_pre_action_pos:
                    # Agent stayed still in a tile that will explode next tick!
                    # This is critically dangerous behavior
                    danger_penalties['imminent_explosion_no_move'] = -5.0
        
        # Track position history for idle detection
        is_new_position = p1_pre_action_pos is None or p1_post_action_pos != p1_pre_action_pos
        if is_new_position:
            self.idle_counter = 0  # Reset idle counter when player moves
        else:
            self.idle_counter += 1  # Increment idle counter when player doesn't move
        
        # For 'new_position' reward: check if current position is new, then update history
        is_new_position_for_reward = self.p1.position not in self.recent_positions
        self.recent_positions.append(self.p1.position) # Maxlen handles history size
        
        # Get observation and calculate reward
        obs = self._obs()
        # Initialize info dict here, before passing to _reward_done
        # Calculate bombs_left for p1
        # Assumes self.p1.id is valid, self.game.bombs contains Bomb objects with a .player_id attribute,
        # and self.game.game_rules["bombs_per_player"] exists.
        p1_active_bombs = sum(1 for b_obj in self.game.bombs if b_obj.player_id == self.p1.id)
        p1_bombs_left_calculated = self.game.game_rules["bombs_per_player"] - p1_active_bombs

        # Calculate bombs_left for p2 if p2 exists
        p2_bombs_left_calculated = None
        if self.p2:
            p2_active_bombs = sum(1 for b_obj in self.game.bombs if b_obj.player_id == self.p2.id)
            p2_bombs_left_calculated = self.game.game_rules["bombs_per_player"] - p2_active_bombs
        
        info = {
            'win': False, # Default to False, _reward_done will update if win/loss
            'rewards': {}, # This will be populated by _reward_done
            'agent_bombed_coords': self.agent_bombed_this_turn_coords,
            'was_in_blast_start_step': self.was_in_blast_at_start_of_turn,
            'is_now_in_blast': self._in_blast(self.p1.position),
            'idle_counter': self.idle_counter,
            'p1_is_new_position': is_new_position_for_reward, # For 'new_position' reward
            'current_frame': self.game.frame,
            'p1_pos': self.p1.position,
            'p2_pos': self.p2.position if self.p2 else None,
            'p1_bombs_left': p1_bombs_left_calculated,
            'p2_bombs_left': p2_bombs_left_calculated,
            'action_mask': self.get_action_mask().tolist(), # For logging
            # 'total_blocks_broken_episode' is now correctly updated before this point
            'total_blocks_broken_episode': self.total_blocks_broken_episode, 
            'entered_imminent_danger': entered_imminent_danger,
            'successful_avoidance': successful_avoidance,
            'in_danger': self._in_blast(self.p1.position),
            **self.game.get_wall_stats(), # Add wall stats (initial_walls, current_walls, walls_broken)
            'chosen_action': action,  # For bump penalty
            'p1_pre_action_pos': p1_pre_action_pos # For bump penalty, captured at start of step
        }
        reward, term = self._reward_done(info) # Pass info to _reward_done, it will be modified in-place
        trunc = False

        # Apply danger penalties with our rebalanced system
        if 'rewards' not in info: info['rewards'] = {}
        
        # We'll store this reward to apply conditionally in _reward_done
        self.successful_avoidance_reward = 0
        if 'successful_avoidance' in danger_penalties:
            self.successful_avoidance_reward = danger_penalties.pop('successful_avoidance')
        
        # Apply all safety-related penalties and rewards
        for penalty_key, penalty_value in danger_penalties.items():
            current_step_scalar_reward += penalty_value  # Can be positive (rewards) or negative (penalties)
            info['rewards'][penalty_key] = penalty_value

        # Apply strategic bomb placement reward
        if strategic_bomb_placement_reward != 0.0:
            current_step_scalar_reward += strategic_bomb_placement_reward
            if 'rewards' not in info: info['rewards'] = {}
            info['rewards']['strategic_bomb_placement'] = strategic_bomb_placement_reward
            
        # Track safety metrics for debugging
        if entered_imminent_danger:
            info['safety_error'] = 'entered_imminent_danger'
        elif successful_avoidance:
            info['safety_success'] = 'moved_to_safety'
        
        # Check for episode timeout
        if not term and self.game.frame >= self.MAX_EPISODE_STEPS:
            term = True  # Terminate the episode
            reward -= 5.0  # Reduced timeout penalty
            info['win'] = False  # Ensure win is false on timeout
            if 'rewards' not in info: info['rewards'] = {} 
            info['rewards']['timeout_penalty'] = -5.0
            
            # Track timeouts for debugging
            self.debug_death_causes["timeout"] += 1
            
            # Log that this episode ended due to timeout
            info['termination_reason'] = 'timeout'
        
        # Apply penalty for repetitive actions
        applied_repetitive_action_penalty = False
        if len(self.action_history) == self.action_history_len:
            # Check if all actions in the history are the same as the first one
            is_repetitive = True
            first_action = self.action_history[0]
            for i in range(1, self.action_history_len):
                if self.action_history[i] != first_action:
                    is_repetitive = False
                    break
            if is_repetitive:
                current_step_scalar_reward += self.penalty_repetitive_action
                applied_repetitive_action_penalty = True
        info['repetitive_action_penalty_applied'] = applied_repetitive_action_penalty

        # Add detailed agent info for evaluation
        info['agent_action'] = action # The action taken by the agent this step
        info['agent_position'] = self.p1.position
        info['agent_active_bombs'] = self._get_player_active_bombs_count(self.p1)
        info['agent_alive'] = self.p1.alive

        return obs, reward, term, trunc, info

    def render(self):
        frame = self.viewer.render_current_frame(save_media=False)
        if self.render_mode == "human":
            import cv2
            cv2.imshow(self.game.id, np.array(frame))
            cv2.waitKey(1)
        return np.asarray(frame)

    def close(self): 
        import cv2; cv2.destroyAllWindows()

    # ---------- helpers ----------
    def _exec(self, player, a):
        # Enforce disable_bombs for curriculum learning
        if hasattr(self, 'disable_bombs') and self.disable_bombs and a == 4: # Action 4 is Bomb
            a = 5 # Change to Still action
        
        # ADDED FAILSAFE: Prevent placing more bombs than allowed
        if a == 4: # If action is Bomb
            # Use the same logic as get_action_mask to determine max bombs
            # MAX_BOMBS is a global constant in this file
            player_max_bombs = self.game.game_rules.get('bombs_per_player', MAX_BOMBS)
            if self._get_player_active_bombs_count(player) >= player_max_bombs:
                a = 5 # Change to Still action if bomb limit reached

        [player.Up, player.Down, player.Left,
         player.Right, player.Bomb, player.Still][a]()

    def _opponent_move(self):
        # Only execute if we have an opponent
        if self.p2 is None:
            return
            
        if self.opponent_style == "none":
            # No opponent action, truly inactive
            self.p2.Still()
        elif self.opponent_style == "still":
            self.p2.Still()
        elif self.opponent_style == "random":
            # Random walker that sometimes places bombs
            a = self.np_random.integers(0,5)
            self._exec(self.p2, a)
        else:  # Default to random
            a = self.np_random.integers(0,5)
            self._exec(self.p2, a)
            
    def set_opponent(self, opponent_type):
        """Set the opponent type for curriculum learning.
        
        Args:
            opponent_type (str): Type of opponent ("still", "random", etc.)
        """
        prev_opponent = self.opponent_style
        self.opponent_style = opponent_type
        print(f"Opponent changed from {prev_opponent} to {opponent_type}")
        return True

    def set_curriculum_stage(self, stage_name: str):
        """Set the curriculum stage for the environment."""
        previous_stage = self.curriculum_stage
        self.curriculum_stage = stage_name
        
        if self.curriculum_stage == 'movement':
            self.disable_bombs = True
        elif self.curriculum_stage == 'bombing':
            self.disable_bombs = False
        else: # Default or None, bombs enabled
            self.disable_bombs = False 
            
        print(f"Curriculum stage changed from {previous_stage} to {self.curriculum_stage}. Bombs disabled: {self.disable_bombs}")
        # The change to disable_bombs will take effect in the _exec method.
        # A reset might be good to ensure a clean state transition if desired.
        return True

    def _obs(self):
        planes = np.zeros((BOARD_H, BOARD_W, 7), np.float32)
        
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                # Plane 0, 1, 2: void, land, block
                tile_type = int(self.game.board[r,c])
                if 0 <= tile_type <= 2:
                    planes[r,c, tile_type] = 1
                
                # Plane 3: Bombs and their timers
                # If a bomb object (marker 1) is present, use its timer from _bomb_dict
                if 1 in self.game.movable_objects[r][c] and self.game._bomb_dict[r][c] > 0:
                    # Normalize timer: game timer is 1-4 (active), 5 (explodes)
                    # Observation: 0.2 (far), 0.4, 0.6, 0.8 (near), 1.0 (exploding this frame via preview)
                    # A bomb with timer 1 (just placed) is 4 ticks away from explosion.
                    # A bomb with timer 4 is 1 tick away from explosion.
                    # Preview markers 2-5 in movable_objects also indicate time to explosion.
                    # Marker 2: 4 ticks away. Marker 5: 1 tick away.
                    # If _bomb_dict shows 1, it's 4 ticks away. If 4, it's 1 tick away.
                    # So, value = (5 - self.game._bomb_dict[r][c]) / 5.0 is not quite right for 'closeness'
                    # Let's use: value = self.game._bomb_dict[r][c] / 5.0. Higher means closer to explosion.
                    planes[r,c,3] = self.game._bomb_dict[r][c] / 5.0

                # Plane 4: Players
                ids = self.game.movable_objects[r][c]
                if -1 in ids: planes[r,c,4] = 1  # P1
                if -2 in ids: planes[r,c,4] = -1 # P2 (or some other value if needed)

                # Plane 5: Danger Zones with CHAIN REACTION AWARENESS
                # First, calculate base danger from markers as before
                danger_value = 0.0
                if 6 in ids: # Actual explosion
                    danger_value = 1.0
                elif 5 in ids: # Explodes next tick
                    danger_value = 0.8
                elif 4 in ids: # Explodes in 2 ticks
                    danger_value = 0.6
                elif 3 in ids: # Explodes in 3 ticks
                    danger_value = 0.4
                elif 2 in ids: # Explodes in 4 ticks
                    danger_value = 0.2
                    
                # Store in planes, but we'll update with chain reaction logic next
                planes[r,c,5] = danger_value

                # Plane 6: Potentially accessible areas (tiles adjacent to blocks)
                # This logic seems fine for identifying tiles next to destructible walls.
                if tile_type == 2: # If this is a block
                    for dr_adj, dc_adj in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nr_adj, nc_adj = r + dr_adj, c + dc_adj
                        if 0 <= nr_adj < BOARD_H and 0 <= nc_adj < BOARD_W and self.game.board[nr_adj, nc_adj] == 1:
                            planes[nr_adj, nc_adj, 6] = 1
        
        # After basic observation is built, update with chain reaction explosion times
        # This ensures that danger zones correctly reflect when explosions will actually happen
        # considering bomb chain reactions (as per design doc)
        chain_explosion_times = self._calculate_chain_reaction_times()
        
        # If we have any chain reaction info, update the danger zones (plane 5)
        if chain_explosion_times:
            for (r, c), time_to_explosion in chain_explosion_times.items():
                # Convert time (1-4) to danger value (0.8-0.2)
                # time=1 → value=0.8 (imminent), time=4 → value=0.2 (distant)
                chain_danger_value = 1.0 - (time_to_explosion * 0.2)
                
                # Only update if the chain reaction makes this tile more dangerous
                # (lower explosion time = higher danger value)
                if chain_danger_value > planes[r, c, 5]:
                    planes[r, c, 5] = chain_danger_value
        return planes

    def _blocks(self):
        """Returns all blocks (destructible walls) in the game
        
        Returns:
            tuple: (row_indices, col_indices) where blocks are located
        """
        return np.where(self.game.board == 2)

    def _void_tiles(self):
        """Returns all void tiles (type 0) in the game
        
        Returns:
            tuple: (row_indices, col_indices) where void tiles are located
        """
        return np.where(self.game.board == 0)

    def _bomb_count(self):
        return sum(1 in self.game.movable_objects[r][c]
                   for r in range(BOARD_H) for c in range(BOARD_W))

    def _blast_tiles(self, r, c):
        blockers = {2}  # wall ids
        tiles = {(r, c)}
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            for k in (1, 2):
                nr, nc = r + dr*k, c + dc*k
                if not (0 <= nr < BOARD_H and 0 <= nc < BOARD_W): break
                if self.game.board[nr, nc] in blockers: break
                tiles.add((nr, nc))
        return tiles
        
    def _calculate_chain_reaction_times(self):
        """Calculate the minimum explosion time for each bomb considering chain reactions.
        
        Returns:
            dict: Mapping from position (r,c) to the minimum time until explosion (1-4 ticks).
                  Lower value means more imminent danger.
        """
        # Initialize explosion times based on markers
        explosion_times = {}
        bomb_positions = []
        
        # First pass: Identify all bombs and their initial explosion times
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                # Check for bomb (marker 1) and get its timer
                if 1 in self.game.movable_objects[r][c] and self.game._bomb_dict[r][c] > 0:
                    # Convert bomb timer (1-4) to ticks until explosion (4-1)
                    # Bomb timer 1 = 4 ticks away, timer 4 = 1 tick away
                    ticks_until_explosion = 5 - self.game._bomb_dict[r][c]
                    bomb_positions.append((r, c))
                    explosion_times[(r, c)] = ticks_until_explosion
        
        # If no bombs, return empty dict
        if not bomb_positions:
            return {}
        
        # Build adjacency graph of which bombs trigger which other bombs
        # A bomb triggers another if the second bomb is in the blast radius of the first
        chain_graph = {}
        for bomb_pos in bomb_positions:
            chain_graph[bomb_pos] = []
            # Calculate blast tiles for this bomb
            blast_tiles = self._blast_tiles(*bomb_pos)
            # Check if any other bombs are in this blast
            for other_bomb in bomb_positions:
                if other_bomb != bomb_pos and other_bomb in blast_tiles:
                    chain_graph[bomb_pos].append(other_bomb)
        
        # Propagate minimum explosion times through the chain graph
        # We'll keep iterating until no more updates are made
        updated = True
        while updated:
            updated = False
            for bomb_pos, triggered_bombs in chain_graph.items():
                current_time = explosion_times[bomb_pos]
                for triggered_bomb in triggered_bombs:
                    # If this bomb causes another to explode sooner than its timer
                    if explosion_times[triggered_bomb] > current_time:
                        explosion_times[triggered_bomb] = current_time
                        updated = True
        
        # Now calculate the explosion time for all affected tiles
        tile_explosion_times = {}
        for bomb_pos in bomb_positions:
            blast_tiles = self._blast_tiles(*bomb_pos)
            min_time = explosion_times[bomb_pos]
            for tile in blast_tiles:
                if tile not in tile_explosion_times or min_time < tile_explosion_times[tile]:
                    tile_explosion_times[tile] = min_time
        
        return tile_explosion_times
    
    def _cache_blast(self):
        """Cache all tiles that are in blast zones (current or future)."""
        self.current_blast = set()
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                # A tile is considered in a blast if it has any preview marker (2-5) or an actual explosion (6)
                if any(marker in self.game.movable_objects[r][c] for marker in [2, 3, 4, 5, 6]):
                    self.current_blast.add((r,c))
    
    def _in_blast(self, pos):
        return pos in self.current_blast
        
    def _get_blast_urgency(self, pos):
        """Get the urgency level of a blast at the given position.
        Higher values mean more imminent explosion.
        
        Args:
            pos: (row, col) position to check
            
        Returns:
            float: Urgency value between 0.0 and 1.0
        """
        if pos not in self.current_blast:
            return 0.0
            
        r, c = pos
        # Check for explosion markers in movable_objects
        # 6: Current explosion, 5: Explodes next tick, 4: Explodes in 2 ticks, etc.
        ids = self.game.movable_objects[r][c]
        if 6 in ids:  # Actual explosion
            return 1.0
        elif 5 in ids:  # Explodes next tick
            return 0.8
        elif 4 in ids:  # Explodes in 2 ticks
            return 0.6
        elif 3 in ids:  # Explodes in 3 ticks
            return 0.4
        elif 2 in ids:  # Explodes in 4 ticks
            return 0.2
        
        # Check active bombs
        if 1 in ids and self.game._bomb_dict[r][c] > 0:
            # Normalize: 1=just placed (0.2), 4=about to explode (0.8)
            # A bomb with timer 1 (just placed) is 4 ticks away from explosion.
            # A bomb with timer 4 is 1 tick away from explosion.
            # Preview markers 2-5 in movable_objects also indicate time to explosion.
            # Marker 2: 4 ticks away. Marker 5: 1 tick away.
            # If _bomb_dict shows 1, it's 4 ticks away. If 4, it's 1 tick away.
            # So, value = (5 - self.game._bomb_dict[r][c]) / 5.0 is not quite right for 'closeness'
            # Let's use: value = self.game._bomb_dict[r][c] / 5.0. Higher means closer to explosion.
            return self.game._bomb_dict[r][c] / 5.0
            
        # Default value for other blast zones
        return 0.3  # Medium urgency
        
    def _get_closest_bomb(self):
        """Find the closest active bomb to the player.
        
        Returns:
            tuple: ((r,c), distance) - Position of closest bomb and Manhattan distance to it,
                  or (None, float('inf')) if no bombs found
        """
        closest_bomb = None
        min_distance = float('inf')
        
        # Get player position
        player_r, player_c = self.p1.position
        
        # Iterate through the bomb dictionary to find all active bombs
        for r in self.game._bomb_dict:
            for c in self.game._bomb_dict[r]:
                # Check if there's an active bomb at this position
                if 1 in self.game.movable_objects[r][c] and self.game._bomb_dict[r][c] > 0:
                    # Calculate Manhattan distance to the bomb
                    distance = abs(player_r - r) + abs(player_c - c)
                    
                    # If this bomb is closer than any we've found so far, update
                    if distance < min_distance:
                        min_distance = distance
                        closest_bomb = (r, c)
        
        return closest_bomb, min_distance
        
    def _calculate_escape_cost(self, pos):
        """Calculate the minimum number of steps needed to escape from a danger zone.
        
        As per design doc:
        - If there's an adjacent safe tile, escape cost is 1
        - For the tile with the bomb itself, escape cost is at least 2
        - If no escape path exists, returns float('inf')
        
        Returns:
            int: Minimum steps to safety (1, 2, etc.) or float('inf') if no escape possible
        """
        if pos not in self.current_blast:
            return 0  # Already safe
            
        # Check immediate neighbors (up, down, left, right)
        r, c = pos
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_r, new_c = r + dr, c + dc
            # Check if neighbor is valid and safe
            if (0 <= new_r < BOARD_H and 0 <= new_c < BOARD_W and 
                self.game.board[new_r, new_c] == 1 and  # Land tile
                (new_r, new_c) not in self.current_blast):  # Not in blast
                return 1  # Can escape in 1 step
                
        # If we're still here, need at least 2 steps to escape
        # Use BFS to find minimum escape path
        from collections import deque
        queue = deque([(r, c, 0)])  # (row, col, distance)
        visited = {(r, c)}
        
        while queue:
            curr_r, curr_c, dist = queue.popleft()
            # Check all 4 directions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_r, new_c = curr_r + dr, curr_c + dc
                if ((new_r, new_c) not in visited and
                    0 <= new_r < BOARD_H and 0 <= new_c < BOARD_W and
                    self.game.board[new_r, new_c] == 1):  # Land tile
                    
                    if (new_r, new_c) not in self.current_blast:
                        return dist + 1  # Found escape path
                        
                    visited.add((new_r, new_c))
                    queue.append((new_r, new_c, dist + 1))
                    
        return float('inf')  # No escape possible

    def _walkable(self, r, c):
        """True if the tile can be stepped on (land only, no blocks or void)."""
        # Check if the tile is land (1) and not occupied by objects
        return (0 <= r < BOARD_H and 0 <= c < BOARD_W and 
                self.game.board[r, c] == 1 and 
                1 not in self.game.movable_objects[r][c])

    def _no_escape(self, r0, c0, simulated_blast):
        """Use time-aware BFS to check if the player has a path to escape the bomb explosion.
        
        Returns True if there's NO escape; False if escape is possible.
        The agent must reach a position OUTSIDE the blast radius before the bomb timer expires.
        """
        from collections import deque
        
        # Use bomb_timer from game_rules
        bomb_timer_duration = 4  # Based on game.py comments: bombs actually explode after 4 ticks. The game_rules['bomb_timer'] is 5, but that seems to be the tick *on which* it explodes (i.e., after 4 full ticks have passed).
        
        # Node = (r, c, t) - position and time after placing the bomb
        q = deque([(r0, c0, 0)])
        vis = {(r0, c0, 0)}
        
        while q:
            r, c, t = q.popleft()
            
            # The key check: We need to be OUTSIDE the blast radius BEFORE the timer expires
            if (r, c) not in simulated_blast and t < bomb_timer_duration:
                return False  # escape found
                
            if t >= bomb_timer_duration - 1:  # stop expanding if we're out of time
                continue
                
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if not self._walkable(nr, nc):
                    continue
                node = (nr, nc, t + 1)
                if node not in vis:
                    vis.add(node)
                    q.append(node)
        
        return True  # no escape

    def get_action_mask(self):
        """Returns a boolean array indicating which actions are allowed (True) or disallowed (False).
        Used by action masking wrapper.
        """
        mask = np.zeros(6, dtype=bool)
        r, c = self.p1.position

        # Action 0: Up
        if self._walkable(r - 1, c):
            mask[0] = True
        # Action 1: Down
        if self._walkable(r + 1, c):
            mask[1] = True
        # Action 2: Left
        if self._walkable(r, c - 1):
            mask[2] = True
        # Action 3: Right
        if self._walkable(r, c + 1):
            mask[3] = True
        
        # Action 4: Bomb
        if self.current_episode_steps == 0:  # Prevent bombing on the very first step of an episode
            mask[4] = False
        else:
            # Player can place a bomb if they haven't reached their limit
            # MAX_BOMBS is a class const, but game_rules might override per player
            player_max_bombs = self.game.game_rules.get('bombs_per_player', MAX_BOMBS) 
            can_place_bomb_limit = self._get_player_active_bombs_count(self.p1) < player_max_bombs
            
            # Curriculum learning check (e.g., disable bombs in early stages)
            disable_bombs_for_curriculum = hasattr(self, 'disable_bombs') and self.disable_bombs
            
            if can_place_bomb_limit and not disable_bombs_for_curriculum:
                # _no_escape check removed for testing: Allow bombing if other conditions (limit, curriculum) are met.
                # The lines for new_bomb_blast and simulated_total_blast are commented out as they were only for _no_escape.
                # # Simulate the blast if a bomb were placed at (r,c)
                # # This includes existing blasts plus the new bomb's potential blast.
                # new_bomb_blast = self._blast_tiles(r, c) # Blast from the potential new bomb
                
                # # self.current_blast should be up-to-date from _cache_blast() called in step() or reset()
                # # It represents blasts from already active bombs on the board.
                # simulated_total_blast = self.current_blast | new_bomb_blast
                
                # if not self._no_escape(r, c, simulated_total_blast):
                #     mask[4] = True  # Bombing is allowed only if there's an escape path
                # # Else, mask[4] remains False (bombing not allowed if _no_escape returns True)
                mask[4] = True # AGENT WILL NOW BOMB IF OTHER CONDITIONS MET (LIMIT, CURRICULUM)
        
        # Action 5: Still - Always allowed
        mask[5] = True
        
        return mask
    
    def _get_player_active_bombs_count(self, player):
        # --- Determine game outcomes ---
        p1_died = not self.p1.alive
        # Ensure p2 exists and check its alive status for win condition
        p1_win = self.p1.alive and (self.p2 is None or not self.p2.alive) 
        game_ended_by_rules = self.game.ended

        # --- Exploration Reward ---
        # Ensure p1_pos is valid and agent is alive before attempting to access/reward exploration
        p1_object = self.game.player_name_to_object.get("P1") # Get P1 object
        if p1_object and p1_object.alive and p1_object.position: # Check P1 object, its alive status, and position
            p1_curr_r, p1_curr_c = p1_object.position # Use P1 object's position
            current_pos_tuple = (p1_curr_r, p1_curr_c)
            if current_pos_tuple not in self.visited_tiles_in_episode:
                current_step_scalar_reward += self.exploration_reward_value
                current_step_reward_breakdown['exploration'] = self.exploration_reward_value
                self.visited_tiles_in_episode.add(current_pos_tuple)
                if self.verbose > 1:
                    print(f"    Exploration reward: +{self.exploration_reward_value} for visiting {current_pos_tuple}")

        # --- Calculate reward components based on old logic ---
        
        # --- Terrain Destruction Rewards ---
        first_time_wall_break_bonus_total = 0.0
        if self.board_state_before_action is not None:
            for r_idx in range(self.BOARD_H):
                for c_idx in range(self.BOARD_W):
                    if self.board_state_before_action[r_idx, c_idx] == 2: # Was a wall
                        if self.game.board[r_idx, c_idx] != 2: # Is no longer a wall
                            # This wall at (r_idx, c_idx) was destroyed this step
                            if (r_idx, c_idx) not in self.destroyed_this_episode_wall_coords:
                                first_time_wall_break_bonus = 50.0 # Increased bonus
                                current_step_scalar_reward += first_time_wall_break_bonus
                                first_time_wall_break_bonus_total += first_time_wall_break_bonus
                                self.destroyed_this_episode_wall_coords.add((r_idx, c_idx))
                                current_step_reward_breakdown.setdefault('first_time_wall_break', 0.0)
                                current_step_reward_breakdown['first_time_wall_break'] += first_time_wall_break_bonus
        if first_time_wall_break_bonus_total > 0:
            pass # Already handled above by setdefault and incrementing

        # 1. Terrain destroyed this step
        current_walls_count = np.sum(self.game.board == 2)
        current_void_count = np.sum(self.game.board == 0)
        
        walls_destroyed_this_step = self.walls_count_before_action - current_walls_count
        voids_cleared_this_step = self.void_count_before_action - current_void_count
        
        terrain_destroyed_reward_value = walls_destroyed_this_step + voids_cleared_this_step
        
        if terrain_destroyed_reward_value > 0:
            general_terrain_destruction_reward = terrain_destroyed_reward_value * 1.0 # General reward for any wall/void cleared
            current_step_scalar_reward += general_terrain_destruction_reward
            current_step_reward_breakdown['general_terrain_destruction'] = current_step_reward_breakdown.get('general_terrain_destruction', 0.0) + general_terrain_destruction_reward
            self.total_blocks_broken_episode += terrain_destroyed_reward_value # Keep updating total for end-of-episode bonus

        # 2. Bombing related rewards/penalties
        any_bomb_placed_by_p1 = self.p1_last_action_was_bomb # Set in step()
        useful_bomb_by_p1 = any_bomb_placed_by_p1 and terrain_destroyed_reward_value > 0
        
        p1_bombed_same_spot = False
        if any_bomb_placed_by_p1 and self.agent_bombed_this_turn_coords:
            p1_active_bombs_now = self._get_player_active_bombs_count(self.p1)
            # If one of P1's bombs exploded this turn, clear recent locations
            # This logic is specific to the old reward system's handling of p1_recent_bomb_locations
            if p1_active_bombs_now < self.p1_active_bombs_count_before_action:
                self.p1_recent_bomb_locations.clear()

            if self.agent_bombed_this_turn_coords in self.p1_recent_bomb_locations:
                p1_bombed_same_spot = True
            
            # Add current bomb location if P1 bombed (after checking for same spot)
            self.p1_recent_bomb_locations.add(self.agent_bombed_this_turn_coords)

        # 3. Idle counter (self.idle_counter is updated in step() and available)

        # 4. Escape
        # General escape (from any blast)
        p1_escaped_blast = self.was_in_blast_at_start_of_turn and not self._in_blast(self.p1.position)
        
        # 5. Distance to opponent
        dist_to_p2 = 0
        if self.p2 and self.p2.alive:
            dist_to_p2 = abs(self.p1.position[0] - self.p2.position[0]) + \
                           abs(self.p1.position[1] - self.p2.position[1])
        
        # 6. New position (exploration)
        # This now comes from the info dict, prepared in step()
        p1_explored_new_tile = info.get('p1_is_new_position', False)

        # --- Assemble Reward ---
        reward = 0.0
        info_rewards = {}

        # Apply rewards/penalties from info['rewards'] set in step() (e.g., strategic_bomb_placement)
        step_rewards = info.get('rewards', {})
        for key, value in step_rewards.items():
            if value != 0:
                current_step_scalar_reward += value
                current_step_reward_breakdown[key] = value
        
        # Apply rewards/penalties from the danger_penalties_dict calculated in step()
        # This dictionary includes items like 'successful_avoidance', 'entered_imminent_danger', etc.
        danger_rewards_and_penalties = info.get('danger_penalties_dict', {})
        for key, value in danger_rewards_and_penalties.items():
            if key == 'successful_avoidance':
                # Successful avoidance reward is only given if the player is not dead and the value is positive.
                if not p1_died and value > 0:
                    current_step_scalar_reward += value
                    current_step_reward_breakdown[key] = value
            elif value != 0: # For other items in the dict (e.g., penalties like 'entered_imminent_danger').
                current_step_scalar_reward += value
                current_step_reward_breakdown[key] = value

        # Reward for bombing a new spot in the current episode
        if self.p1_last_action_was_bomb and self.agent_bombed_this_turn_coords:
            if self.agent_bombed_this_turn_coords not in self.bombed_locations_in_episode:
                new_bomb_spot_reward_value = 25.0
                current_step_scalar_reward += new_bomb_spot_reward_value
                current_step_reward_breakdown['new_bomb_spot_reward'] = new_bomb_spot_reward_value
                self.bombed_locations_in_episode.add(self.agent_bombed_this_turn_coords)
                if self.verbose > 0:
                    print(f"    Reward: New bomb spot at {self.agent_bombed_this_turn_coords}, reward: {new_bomb_spot_reward_value}")

        # Penalty for non-strategic bomb placement
        if self.p1_last_action_was_bomb and self.agent_bombed_this_turn_coords:
            bomb_r, bomb_c = self.agent_bombed_this_turn_coords
            bomb_power = self.p1.power # Player's current bomb power

            # Tile types (integer literals based on game constants)
            # TILE_VOID = 0
            # TILE_WALL = 2 (destructible)
            # TILE_PERMANENT = 3 (indestructible)

            is_strategic_bomb = False # True if hits wall, void, or opponent

            blast_coords = set()
            blast_coords.add((bomb_r, bomb_c)) # Bomb location itself

            # Horizontal blast
            # Right
            for i in range(1, bomb_power + 1):
                br, bc = bomb_r, bomb_c + i
                if 0 <= bc < self.BOARD_W:
                    blast_coords.add((br, bc))
                    if self.board_state_before_action is not None and self.board_state_before_action[br, bc] == 3: # TILE_PERMANENT
                        break 
                else: break # Out of bounds
            # Left
            for i in range(1, bomb_power + 1):
                br, bc = bomb_r, bomb_c - i
                if 0 <= bc < self.BOARD_W:
                    blast_coords.add((br, bc))
                    if self.board_state_before_action is not None and self.board_state_before_action[br, bc] == 3: # TILE_PERMANENT
                        break
                else: break
            
            # Vertical blast
            # Down
            for i in range(1, bomb_power + 1):
                br, bc = bomb_r + i, bomb_c
                if 0 <= br < self.BOARD_H:
                    blast_coords.add((br, bc))
                    if self.board_state_before_action is not None and self.board_state_before_action[br, bc] == 3: # TILE_PERMANENT
                        break
                else: break
            # Up
            for i in range(1, bomb_power + 1):
                br, bc = bomb_r - i, bomb_c
                if 0 <= br < self.BOARD_H:
                    blast_coords.add((br, bc))
                    if self.board_state_before_action is not None and self.board_state_before_action[br, bc] == 3: # TILE_PERMANENT
                        break
                else: break

            # Check if opponent is in blast radius
            if self.p2 and self.p2.alive and hasattr(self.p2, 'position') and self.p2.position in blast_coords:
                is_strategic_bomb = True

            hits_wall_or_void = False # New flag specifically for terrain
            # Check if blast hits walls or void using board_state_before_action
            if self.board_state_before_action is not None:
                for r_blast, c_blast in blast_coords:
                    if not (0 <= r_blast < self.BOARD_H and 0 <= c_blast < self.BOARD_W):
                        continue
                    
                    tile_type_before = self.board_state_before_action[r_blast, c_blast]
                    if tile_type_before == 2: # TILE_WALL (destructible)
                        is_strategic_bomb = True
                        hits_wall_or_void = True # Hits terrain
                        # Don't break here if we also want to check for opponent in blast_coords later
                    elif tile_type_before == 0: # TILE_VOID
                        is_strategic_bomb = True
                        hits_wall_or_void = True # Hits terrain
                        # Don't break here
            
            # Re-check opponent in blast if not already strategic from terrain
            if not is_strategic_bomb and self.p2 and self.p2.alive and hasattr(self.p2, 'position') and self.p2.position in blast_coords:
                is_strategic_bomb = True

            # Apply penalty if bomb is NOT strategic (hits nothing useful)
            if not is_strategic_bomb:
                non_strategic_bomb_penalty_value = -75.0
                current_step_reward_breakdown['non_strategic_bomb_penalty'] = non_strategic_bomb_penalty_value
                if self.verbose > 0:
                    print(f"    Penalty: Non-strategic bomb at {(bomb_r, bomb_c)}, penalty: {non_strategic_bomb_penalty_value}")
            # Apply reward if bomb is aimed at terrain
            elif hits_wall_or_void: # Changed from 'if hits_wall_or_void:' to 'elif hits_wall_or_void:' to avoid double reward if also non_strategic (which shouldn't happen with current logic)
                strategic_terrain_bomb_reward = 15.0
                current_step_scalar_reward += strategic_terrain_bomb_reward
                current_step_reward_breakdown['strategic_terrain_bomb_reward'] = strategic_terrain_bomb_reward
                if self.verbose > 0:
                    print(f"    Reward: Strategic terrain bomb at {(bomb_r, bomb_c)}, reward: {strategic_terrain_bomb_reward}")
                
        # Positive Rewards
        # Base reward for breaking walls (from +10 to +50)
        base_wall_reward = 50 * terrain_destroyed_reward_value
        
        # EXPONENTIAL reward for breaking walls - making each additional wall much more valuable
        # If walls were broken this step, add an exponential bonus based on total broken so far
        if not p1_died and terrain_destroyed_reward_value > 0:
            # The more walls you've broken so far, the bigger the reward for new walls
            # This makes breaking the 5th wall much more valuable than the 1st
            exponential_factor = 1.5 ** self.total_blocks_broken_episode
            wall_reward = base_wall_reward * exponential_factor
            current_step_scalar_reward += wall_reward
            current_step_reward_breakdown['terrain_destroyed'] = wall_reward
            
            # Add a big immediate bonus for breaking multiple walls in one action
            # This encourages bomb placement that can break multiple walls at once
            if terrain_destroyed_reward_value >= 2:
                multi_wall_bonus = terrain_destroyed_reward_value * 100  # 200+ points for breaking 2+ walls at once
                current_step_scalar_reward += multi_wall_bonus
                current_step_reward_breakdown['multi_wall_bonus'] = multi_wall_bonus
        # else: # This block was redundant as base_wall_reward is 0 if terrain_destroyed_reward_value is 0
        #     # Add base reward with no bonus for non-wall-breaking steps
        #     current_step_scalar_reward += base_wall_reward
        
        if p1_win:
            current_step_scalar_reward += 50
            current_step_reward_breakdown['win'] = 50
            
        # Reward escape if player didn't die, regardless of wall breaking
        if not p1_died:
            # Successful Dodge Reward (if player moved out of an exploding tile and survived)
            if p1_escaped_blast: # p1_escaped_blast is: self.was_in_blast_at_start_of_turn and not self._in_blast(self.p1.position)
                successful_dodge_reward_value = 10.0
                current_step_scalar_reward += successful_dodge_reward_value
                current_step_reward_breakdown['successful_dodge_reward'] = successful_dodge_reward_value
                if self.verbose > 0:
                    print(f"    Reward: Successful dodge, reward: {successful_dodge_reward_value}")
                
            # NEW: Give movement guidance when in danger
            # If player is currently in danger, reward moving AWAY from bombs
            if self._in_blast(self.p1.position):
                # Find the closest bomb
                closest_bomb_pos, closest_bomb_dist = self._get_closest_bomb()
                if closest_bomb_pos:
                    r1, c1 = self.p1.position
                    r2, c2 = closest_bomb_pos
                    
                    # Check if player moved away from bomb in this step
                    if len(self.pos_history) >= 2:
                        prev_r, prev_c = self.pos_history[-2]
                        prev_dist = abs(prev_r - r2) + abs(prev_c - c2)  # Manhattan distance
                        curr_dist = abs(r1 - r2) + abs(c1 - c2)  # Manhattan distance
                        
                        if curr_dist > prev_dist:  # Player moved AWAY from bomb
                            away_reward = 30 * self._get_blast_urgency(self.p1.position)  # Scale with urgency
                            current_step_scalar_reward += away_reward
                            current_step_reward_breakdown['moving_away_from_bomb'] = away_reward
            
        distance_reward = 8 * max(0, 10 - dist_to_p2)
        current_step_scalar_reward += distance_reward
        if distance_reward != 0: current_step_reward_breakdown['distance_to_opponent'] = distance_reward

        if p1_explored_new_tile:
            current_step_scalar_reward += 30  # Increased from 10 to 30 to encourage exploration
            current_step_reward_breakdown['new_position'] = 30

        # Negative Rewards (Penalties)
        # Per design doc: "Placing bomb at same place - 2"
        if p1_bombed_same_spot:
            reward -= 5.0
            current_step_reward_breakdown['same_spot_bomb'] = -5.0

        # --- New Bump Penalty --- 
        chosen_action_for_bump = info.get('chosen_action')
        p1_pre_action_pos_for_bump = info.get('p1_pre_action_pos')
        bump_penalty_applied_this_step = False

        # DEBUG: Print values for bump penalty check
        if self.verbose > 1: # Only print if verbose level is high enough
            print(f"[DEBUG _reward_done] Frame: {self.game.frame}, ChosenAction: {chosen_action_for_bump}, PreActionPos: {p1_pre_action_pos_for_bump}, CurrentPos: {self.p1.position}, P1Died: {p1_died}")

        # Actions 0-3 are Up, Down, Left, Right
        if chosen_action_for_bump is not None and 0 <= chosen_action_for_bump <= 3:
            if p1_pre_action_pos_for_bump is not None and \
               p1_pre_action_pos_for_bump == self.p1.position and \
               not p1_died: # Don't apply bump penalty if agent died (death penalty is enough)
                # Agent attempted a movement action but didn't change position
                bump_penalty_val = -10.0  # Penalty for bumping into a wall/obstacle
                current_step_scalar_reward += bump_penalty_val
                current_step_reward_breakdown['bump_penalty'] = bump_penalty_val
                bump_penalty_applied_this_step = True
        # --- End New Bump Penalty ---
            
        # Per design doc: "Dying - 20"
        if p1_died:
            base_death_penalty = -100.0 # Increased death penalty
            current_step_scalar_reward += base_death_penalty
            current_step_reward_breakdown['death_base'] = base_death_penalty
            
            # Track death for debugging
            self.debug_episode_deaths += 1
            self.debug_avg_survival_time = ((self.debug_avg_survival_time * (self.debug_episode_deaths - 1)) + 
                                           self.current_episode_steps) / self.debug_episode_deaths
            
            # Track death cause - with proper attribution
            # In this simplified 2-player setup, with a still/none opponent,
            # all deaths are from the agent's own bombs
            self.debug_death_causes["self_bomb"] += 1
            reward -= 100.0  # Significantly increased penalty
            current_step_reward_breakdown['death_self_bomb'] = -100.0
        # Only penalize for staying still for 3+ turns in a row
        # Adjusted to not penalize for idle if a bump penalty was just applied for the same non-movement
        idle_penalty_val = 0
        if self.idle_counter >= 2:  # 0-indexed, so 2 means 3 frames without moving
            if not bump_penalty_applied_this_step: # Only apply idle if not due to a penalized bump
                idle_penalty_val = -3
                current_step_scalar_reward += idle_penalty_val
                current_step_reward_breakdown['idle_penalty'] = idle_penalty_val
        
        # Termination condition
        # 'term' might have already been set by timeout logic earlier in step(), or by game rules here.
        # This re-evaluation ensures it's correctly set based on player/game status before returning.
        term = p1_died or p1_win or game_ended_by_rules
        
        # Penalty for walls remaining at the end of the episode
        if term: # Apply only if the episode is truly ending
            walls_remaining = np.sum(self.game.board == 2) # Count destructible walls
            remaining_wall_penalty = -20.0 * walls_remaining
            current_step_scalar_reward += remaining_wall_penalty
            current_step_reward_breakdown['remaining_wall_penalty'] = remaining_wall_penalty # Simplified update
            if self.verbose > 0:
                print(f"    End of Episode: Walls Remaining: {walls_remaining}, Penalty: {remaining_wall_penalty}")

        # Clipping
        reward = float(np.clip(reward, -120, 300)) # Increased upper clip range
        

        # Update the info dict's rewards with those calculated in this function
        # All reward components for this step are now in current_step_reward_breakdown.
        # Assign this to info['rewards'] to be passed out.
        info['rewards'] = current_step_reward_breakdown
        info['win'] = p1_win # Ensure info reflects the win status
        return current_step_scalar_reward, info
