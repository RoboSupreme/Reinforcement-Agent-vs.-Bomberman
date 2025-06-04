import numpy as np
from uuid import uuid4 as random_id_generator
from functools import wraps


class BombObject:
    def __init__(self, r, c, timer, player_id, blast_radius):
        self.r = r
        self.c = c
        self.position = (r, c)
        self.timer = timer # Initial timer ticks (e.g., 4)
        self.player_id = player_id
        self.blast_radius = blast_radius
        self.exploded = False # To mark if it has been processed for explosion

    def __repr__(self):
        return f"Bomb(pos=({self.r},{self.c}), timer={self.timer}, player_id={self.player_id})"


class Game:

    def __init__(self, map_scheme, verbose=False):
        '''Initialize the game.'''
        self.verbose = verbose

        ## initialize the map scheme ##
        self.map_scheme = map_scheme

        ## assign a random ID to the game ##
        self.id = str(random_id_generator())

        ## start frame ##
        self.frame = 0
        self.ended = False

        map_type = self.map_scheme["name"]
        if map_type == 'standard':
            ## initialize board with the specified dimensions ##
            self.board = np.zeros((7, 11))
            # movable objects is a dictionary of lists with the following format: {}[row][column] = [objects]
            self.movable_objects = dict()
            self._bomb_dict = dict()
            for row in range(self.board.shape[0]):
                self.movable_objects[row] = dict()
                self._bomb_dict[row] = dict()
                for column in range(self.board.shape[1]):
                    self._bomb_dict[row][column] = 0
                    self.movable_objects[row][column] = set()

            ## add starting floor to the map ##
            # upper left
            self.board[0, 0], self.board[0, 1], self.board[1, 0] = 1, 1, 1
            # upper right
            self.board[0, -1], self.board[0, -2], self.board[1, -1] = 1, 1, 1
            # bottom left
            self.board[-1, 0], self.board[-2, 0], self.board[-1, 1] = 1, 1, 1
            # bottom right
            self.board[-1, -1], self.board[-2, -1], self.board[-1, -2] = 1, 1, 1

            ## add hard walls ##
            self.board[1, 1], self.board[2, 2], self.board[2, 1] = 2, 2, 2
            self.board[-2, 1], self.board[-3, 1], self.board[-3, 2] = 2, 2, 2
            self.board[-2, -2], self.board[-3, -3], self.board[-3, -2] = 2, 2, 2
            self.board[1, -2], self.board[2, -3], self.board[2, -2] = 2, 2, 2

            self.board[0, 3], self.board[0, -4] = 2, 2
            self.board[-1, 3], self.board[-1, -4] = 2, 2

            self.board[2, 5], self.board[3, 5], self.board[4, 5], self.board[3, 4], self.board[3, -5] = 2, 2, 2, 2, 2

        elif map_type == 'IBM':
            ## initialize board with the specified dimensions ##
            self.board = np.zeros((15, 22))
            # movable objects is a dictionary of lists with the following format: {}[row][column] = [objects]
            self.movable_objects = dict()
            self._bomb_dict = dict()
            for row in range(self.board.shape[0]):
                self.movable_objects[row] = dict()
                self._bomb_dict[row] = dict()
                for column in range(self.board.shape[1]):
                    self._bomb_dict[row][column] = 0
                    self.movable_objects[row][column] = set()

            ## add starting floor to the map ##
            # upper left
            self.board[0, 0], self.board[0, 1], self.board[1, 0] = 1, 1, 1
            # upper right
            self.board[0, -1], self.board[0, -2], self.board[1, -1] = 1, 1, 1
            # bottom left
            self.board[-1, 0], self.board[-2, 0], self.board[-1, 1] = 1, 1, 1
            # bottom right
            self.board[-1, -1], self.board[-2, -1], self.board[-1, -2] = 1, 1, 1

            ## add hard walls ##
            # I
            self.board[3, 2], self.board[3, 3], self.board[5, 2], self.board[5, 3] = 2, 2, 2, 2
            self.board[7, 2], self.board[7, 3], self.board[9, 2], self.board[9, 3] = 2, 2, 2, 2
            self.board[11, 2], self.board[11, 3] = 2, 2

            # B
            self.board[3, 6], self.board[3, 7], self.board[3, 9] = 2, 2, 2
            self.board[5, 7], self.board[5, 10] = 2, 2
            self.board[7, 7], self.board[7, 9] = 2, 2
            self.board[9, 7], self.board[9, 10] = 2, 2
            self.board[11, 6], self.board[11, 7], self.board[11, 9] = 2, 2, 2

            # M
            self.board[3, 13], self.board[3, 15], self.board[3, 17], self.board[3, 19] = 2, 2, 2, 2
            self.board[5, 13], self.board[5, 16], self.board[5, 19] = 2, 2, 2
            self.board[7, 13], self.board[7, 16], self.board[7, 19] = 2, 2, 2
            self.board[9, 13], self.board[9, 19] = 2, 2
            self.board[11, 13], self.board[11, 19] = 2, 2

        ## define player starting positions (this also defines how many players are available) ##
        self.player_name_to_object = dict()
        # 2 players
        self.player_slots = [(0, 0), (self.board.shape[0] - 1, self.board.shape[1] - 1)]
        self.players = []
        self.player_action_queue = dict()
        self.bombs = [] # Initialize list to store active Bomb objects

        # Count initial destructible walls (soft and hard)
        self.initial_wall_count = np.count_nonzero(self.board == 2) + np.count_nonzero(self.board == 0)

        # Define game rules
        self.game_rules = {
            "bombs_per_player": 2,  # NOTE: This rule is NOT CURRENTLY ENFORCED in the code. Players can place multiple bombs.
            "bomb_blast_radius": 2, # Effective blast radius is 2 tiles (due to function default parameters).
            "bomb_timer": 5          # NOTE: This value (10) is NOT CURRENTLY USED for bomb countdown. Bombs actually explode after 4 ticks (when _bomb_dict value reaches 5).
        }

    def _add_player(self, player_instance):
        assert player_instance.name not in [p.name for p in
                                            self.players], 'There is already a player with the name {}'.format(
            player_instance.name)
        assert len(self.players) < len(self.player_slots), 'All players are already loaded in.'

        # append player to player list
        self.players.append(player_instance)

        # assign id to the player
        player_instance.id = -(len(self.players))

        # update dictionaries
        self.player_name_to_object[player_instance.name] = player_instance

        # assign starting position to the player
        player_instance.position = self.player_slots[len(self.players) - 1]
        self.movable_objects[player_instance.position[0]][player_instance.position[1]].update([player_instance.id])

        # initialize actions for the player
        self.player_action_queue[player_instance.name] = None

    def _possible_move(self, next_location_r, next_location_c):
        # if we are at the edge of the map
        if next_location_r not in range(self.board.shape[0]) or next_location_c not in range(self.board.shape[1]):
            return False
        # if the next location is a Soft Wall (impassable)
        if self.board[next_location_r, next_location_c] == 0:
            return False
        # if the next location is a Hard Wall (impassable)
        if self.board[next_location_r, next_location_c] == 2:
            return False
        # if the next location is a bomb
        if 1 in self.movable_objects[next_location_r][next_location_c]:
            return False
        # if no blocks, return True
        return True

    def get_wall_stats(self):
        """Counts initial Soft/Hard Walls, current Soft/Hard Walls, and calculates broken walls."""
        current_walls = np.count_nonzero(self.board == 2) + np.count_nonzero(self.board == 0)
        walls_broken = self.initial_wall_count - current_walls
        return {
            "initial_walls": self.initial_wall_count,
            "current_walls": current_walls,
            "walls_broken": walls_broken
        }

    def start(self):
        assert len(self.players) == len(self.player_slots), 'Not all players are loaded in yet.'
        assert self.frame == 0, 'The game is already ongoing.'
        self.frame = 1
        if self.verbose:
            print('The game has been started! (id: {})'.format(self.id))
        return True

    def _move_player(self, new_r, new_c, player):
        # remove player from old position
        self.movable_objects[player.position[0]][player.position[1]].remove(player.id)
        # update player position to new position
        self.movable_objects[new_r][new_c].update([player.id])
        player.position = (new_r, new_c)

    def get_predicted_blast_coords(self, bomb_r, bomb_c, blast_range):
        """Predicts the blast coordinates of a bomb without changing game state.

        Args:
            bomb_r (int): Row of the bomb.
            bomb_c (int): Column of the bomb.
            blast_range (int): Blast radius of the bomb.

        Returns:
            set: A set of (r,c) tuples representing all tiles affected by the blast.
        """
        affected_coords = set()
        affected_coords.add((bomb_r, bomb_c)) # Bomb's own location is affected

        # Calculate blast in all four directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Right, Left, Down, Up
            for i in range(1, blast_range + 1):
                r, c = bomb_r + dr * i, bomb_c + dc * i

                if not (0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]):
                    break # Out of bounds

                affected_coords.add((r, c))

                if self.board[r, c] == 2: # Hard Wall
                    break # Explosion stops at hard walls
                # Soft walls (0) are destroyed but do not stop the blast path for prediction
        return affected_coords

    def _explode_bomb(self, bomb_r, bomb_c, blast_range=2):
        """Explode a bomb at the given position, calculating its blast area.
        
        Args:
            bomb_r (int): Row of the bomb to explode
            bomb_c (int): Column of the bomb to explode
            blast_range (int): Blast radius
            
        Returns:
            list: List of (r,c) positions of other bombs triggered by this explosion
        """
        # print(f"[DEBUG] Exploding bomb at ({bomb_r}, {bomb_c})")
        
        # Clean up the bomb marker if it exists
        if 1 in self.movable_objects[bomb_r][bomb_c]:
            self.movable_objects[bomb_r][bomb_c].remove(1)
        self._bomb_dict[bomb_r][bomb_c] = 0  # Reset bomb timer

        # Remove the BombObject from self.bombs
        # Iterate backwards if removing to avoid index issues, or build a new list
        # For simplicity here, we find and remove the first match. 
        # This assumes one bomb per tile, which is current game logic.
        for i, bomb_obj in enumerate(self.bombs):
            if bomb_obj.position == (bomb_r, bomb_c) and not bomb_obj.exploded:
                bomb_obj.exploded = True # Mark as exploded to prevent re-processing if logic changes
                # self.bombs.pop(i) # This would be one way, but safer to rebuild or filter
                break # Assuming only one bomb object per tile to explode at a time
        # Rebuild self.bombs to exclude any that were marked as exploded in this frame's logic
        # This handles chain reactions where multiple bombs might be marked 'exploded' before this list comprehension runs.
        self.bombs = [b for b in self.bombs if not b.exploded or b.timer > 0] # Keep bombs that are not exploded OR if their timer is still >0 (e.g. if a different system handles timer decrement)
        # A simpler approach if not handling complex chain reaction state within BombObject itself for removal:
        # self.bombs = [b for b in self.bombs if not (b.position == (bomb_r, bomb_c) and b.timer <= 0)]
        # For now, let's stick to a more direct removal based on the current bomb being exploded:
        # This needs to be robust. If multiple bombs explode in the same frame due to chain reaction,
        # _explode_bomb might be called for each. Marking `exploded=True` helps.
        # The list comprehension `self.bombs = [b for b in self.bombs if not b.exploded]` should ideally be at the end of all explosion processing for a frame.
        # For now, let's assume _explode_bomb is the primary point of removal for a single bomb event.
        
        # Find the specific bomb and remove it
        bomb_to_remove = None
        for bomb_obj in self.bombs:
            if bomb_obj.position == (bomb_r, bomb_c):
                bomb_to_remove = bomb_obj
                break
        if bomb_to_remove:
            self.bombs.remove(bomb_to_remove)
        
        # Clear previews for this specific bomb first
        self._clear_specific_bomb_previews(bomb_r, bomb_c, blast_range)

        # Make the center position explode
        self.movable_objects[bomb_r][bomb_c].add(6)  # Add explosion
        self.board[bomb_r, bomb_c] = 1  # Ensure center is floor
        
        # Track triggered bombs for chain reactions
        triggered_bombs = []
        
        # Calculate blast in all four directions
        advance_r_plus = True
        advance_r_min = True
        advance_c_plus = True
        advance_c_min = True
        
        for d in range(1, blast_range + 1):  # Start from 1 to skip center
            # DOWN
            if (bomb_r + d) in range(self.board.shape[0]) and advance_r_plus:
                r, c = bomb_r + d, bomb_c
                # Check for bombs here
                if 1 in self.movable_objects[r][c] and self._bomb_dict[r][c] > 0:
                    triggered_bombs.append((r, c))
                    
                # Mark explosion
                self.movable_objects[r][c].add(6)  # Add explosion
                
                # Check for Hard Walls and Soft Walls
                if self.board[r, c] == 2:  # Hard Wall - becomes Floor and stops explosion
                    # print(f"[DEBUG] Hard Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert to floor
                    advance_r_plus = False  # Stop in this direction
                elif self.board[r, c] == 0:  # Soft Wall - becomes Floor and explosion continues
                    # print(f"[DEBUG] Soft Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert Soft Wall to floor
                    # Don't set advance_r_plus = False so explosion continues
            
            # UP
            if (bomb_r - d) in range(self.board.shape[0]) and advance_r_min:
                r, c = bomb_r - d, bomb_c
                # Check for bombs
                if 1 in self.movable_objects[r][c] and self._bomb_dict[r][c] > 0:
                    triggered_bombs.append((r, c))
                    
                # Mark explosion
                self.movable_objects[r][c].add(6)  # Add explosion
                
                # Check for Hard Walls and Soft Walls
                if self.board[r, c] == 2:  # Hard Wall - becomes Floor and stops explosion
                    # print(f"[DEBUG] Hard Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert to floor
                    advance_r_min = False  # Stop in this direction
                elif self.board[r, c] == 0:  # Soft Wall - becomes Floor and explosion continues
                    # print(f"[DEBUG] Soft Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert Soft Wall to floor
                    # Don't set advance_r_min = False so explosion continues
            
            # RIGHT
            if (bomb_c + d) in range(self.board.shape[1]) and advance_c_plus:
                r, c = bomb_r, bomb_c + d
                # Check for bombs
                if 1 in self.movable_objects[r][c] and self._bomb_dict[r][c] > 0:
                    triggered_bombs.append((r, c))
                    
                # Mark explosion
                self.movable_objects[r][c].add(6)  # Add explosion
                
                # Check for Hard Walls and Soft Walls
                if self.board[r, c] == 2:  # Hard Wall - becomes Floor and stops explosion
                    # print(f"[DEBUG] Hard Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert to floor
                    advance_c_plus = False  # Stop in this direction
                elif self.board[r, c] == 0:  # Soft Wall - becomes Floor and explosion continues
                    # print(f"[DEBUG] Soft Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert Soft Wall to floor
                    # Don't set advance_c_plus = False so explosion continues
            
            # LEFT
            if (bomb_c - d) in range(self.board.shape[1]) and advance_c_min:
                r, c = bomb_r, bomb_c - d
                # Check for bombs
                if 1 in self.movable_objects[r][c] and self._bomb_dict[r][c] > 0:
                    triggered_bombs.append((r, c))
                    
                # Mark explosion
                self.movable_objects[r][c].add(6)  # Add explosion
                
                # Check for Hard Walls and Soft Walls
                if self.board[r, c] == 2:  # Hard Wall - becomes Floor and stops explosion
                    # print(f"[DEBUG] Hard Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert to floor
                    advance_c_min = False  # Stop in this direction
                elif self.board[r, c] == 0:  # Soft Wall - becomes Floor and explosion continues
                    # print(f"[DEBUG] Soft Wall at ({r}, {c}) destroyed by bomb at ({bomb_r}, {bomb_c})")
                    self.board[r, c] = 1  # Convert Soft Wall to floor
                    # Don't set advance_c_min = False so explosion continues
        
        return triggered_bombs
    
    def _check_bomb_chain_reaction(self, explosion_r, explosion_c, blast_range=2):
        """Check for bombs in the explosion radius and trigger them in a non-recursive way.
        
        Args:
            explosion_r (int): Row of the exploding bomb
            explosion_c (int): Column of the exploding bomb
            blast_range (int): Blast radius for bombs
        """
        # Queue of bombs that need to explode
        bomb_queue = self._explode_bomb(explosion_r, explosion_c, blast_range)
        
        # Process all bombs in the queue (any new bombs get added to the queue)
        while bomb_queue:
            # Get the next bomb to process
            bomb_r, bomb_c = bomb_queue.pop(0)
            # print(f"[DEBUG] CHAIN REACTION: Processing bomb at ({bomb_r}, {bomb_c})")
            
            # Explode this bomb and get any new bombs it triggers
            new_triggered_bombs = self._explode_bomb(bomb_r, bomb_c, blast_range)
            
            # Add any new triggered bombs to the queue
            for new_bomb in new_triggered_bombs:
                if new_bomb not in bomb_queue:  # Avoid duplicates
                    bomb_queue.append(new_bomb)
                    # print(f"[DEBUG] Added bomb at {new_bomb} to explosion queue")
        
        # One final cleanup after all chain reactions are processed
        self._clean_all_orphaned_preview_markers()
        # print(f"[DEBUG] Chain reaction complete, processed all bombs")
        return
    
    def _wipe_preview_markers(self, r, c):
        """Remove stage‑2…5 preview markers so they don't keep ticking."""
        # Ensure the keys exist before trying to update
        if r in self.movable_objects and c in self.movable_objects[r]:
            self.movable_objects[r][c].difference_update({2, 3, 4, 5})

    def _clear_specific_bomb_previews(self, bomb_r, bomb_c, blast_range=2):
        """Clears all preview markers (2-5) for a specific bomb that is about to explode."""
        # Central tile
        if bomb_r in self.movable_objects and bomb_c in self.movable_objects[bomb_r]:
            self.movable_objects[bomb_r][bomb_c].difference_update({2, 3, 4, 5})

        advance_r_plus = True
        advance_r_min = True
        advance_c_plus = True
        advance_c_min = True

        for d in range(1, blast_range + 1): # Start from 1 to clear outward paths
            # DOWN
            if (bomb_r + d) in range(self.board.shape[0]) and advance_r_plus:
                r, c = bomb_r + d, bomb_c
                if r in self.movable_objects and c in self.movable_objects[r]:
                    self.movable_objects[r][c].difference_update({2, 3, 4, 5})
                if self.board[r, c] == 2: # Wall
                    advance_r_plus = False
            
            # UP
            if (bomb_r - d) in range(self.board.shape[0]) and advance_r_min:
                r, c = bomb_r - d, bomb_c
                if r in self.movable_objects and c in self.movable_objects[r]:
                    self.movable_objects[r][c].difference_update({2, 3, 4, 5})
                if self.board[r, c] == 2: # Wall
                    advance_r_min = False
            
            # RIGHT
            if (bomb_c + d) in range(self.board.shape[1]) and advance_c_plus:
                r, c = bomb_r, bomb_c + d
                if r in self.movable_objects and c in self.movable_objects[r]:
                    self.movable_objects[r][c].difference_update({2, 3, 4, 5})
                if self.board[r, c] == 2: # Wall
                    advance_c_plus = False
            
            # LEFT
            if (bomb_c - d) in range(self.board.shape[1]) and advance_c_min:
                r, c = bomb_r, bomb_c - d
                if r in self.movable_objects and c in self.movable_objects[r]:
                    self.movable_objects[r][c].difference_update({2, 3, 4, 5})
                if self.board[r, c] == 2: # Wall
                    advance_c_min = False

    def _refresh_all_active_bomb_previews(self, default_blast_range=2):
        """Refreshes preview markers for all currently active and ticking bombs."""
        for r_bomb_loc in self._bomb_dict:
            for c_bomb_loc in self._bomb_dict[r_bomb_loc]:
                timer_value = self._bomb_dict[r_bomb_loc][c_bomb_loc]
                if 1 <= timer_value <= 4: # Bomb is active and ticking
                    correct_preview_id = timer_value + 1
                    
                    # Calculate blast path for this active bomb
                    # Central tile first
                    if r_bomb_loc in self.movable_objects and c_bomb_loc in self.movable_objects[r_bomb_loc]:
                        self.movable_objects[r_bomb_loc][c_bomb_loc].update([correct_preview_id])

                    advance_r_plus = True
                    advance_r_min = True
                    advance_c_plus = True
                    advance_c_min = True

                    for d in range(1, default_blast_range + 1):
                        # DOWN
                        if (r_bomb_loc + d) in range(self.board.shape[0]) and advance_r_plus:
                            pr, pc = r_bomb_loc + d, c_bomb_loc
                            if pr in self.movable_objects and pc in self.movable_objects[pr]:
                                self.movable_objects[pr][pc].update([correct_preview_id])
                            if self.board[pr, pc] == 2: # Wall
                                advance_r_plus = False
                        
                        # UP
                        if (r_bomb_loc - d) in range(self.board.shape[0]) and advance_r_min:
                            pr, pc = r_bomb_loc - d, c_bomb_loc
                            if pr in self.movable_objects and pc in self.movable_objects[pr]:
                                self.movable_objects[pr][pc].update([correct_preview_id])
                            if self.board[pr, pc] == 2: # Wall
                                advance_r_min = False
                        
                        # RIGHT
                        if (c_bomb_loc + d) in range(self.board.shape[1]) and advance_c_plus:
                            pr, pc = r_bomb_loc, c_bomb_loc + d
                            if pr in self.movable_objects and pc in self.movable_objects[pr]:
                                self.movable_objects[pr][pc].update([correct_preview_id])
                            if self.board[pr, pc] == 2: # Wall
                                advance_c_plus = False
                        
                        # LEFT
                        if (c_bomb_loc - d) in range(self.board.shape[1]) and advance_c_min:
                            pr, pc = r_bomb_loc, c_bomb_loc - d
                            if pr in self.movable_objects and pc in self.movable_objects[pr]:
                                self.movable_objects[pr][pc].update([correct_preview_id])
                            if self.board[pr, pc] == 2: # Wall
                                advance_c_min = False

    def _clean_all_orphaned_preview_markers(self):
        """Scan the entire board and remove any orphaned preview markers (2-5).
        This ensures no preview markers continue ticking after bombs are triggered.
        """
        # Get all positions with preview markers
        for r in self.movable_objects.keys():
            for c in list(self.movable_objects[r].keys()):
                # Get the set of markers at this position
                marker_set = self.movable_objects[r][c]
                
                # Get preview markers (stages 2-5)
                preview_markers = marker_set.intersection({2, 3, 4, 5})
                
                # If this tile has preview markers, but no active bomb (1) is at this exact cell,
            # AND no current explosion (6) is at this cell (which might be temporarily obscuring a valid preview),
            # then the preview markers are considered orphaned.
            if preview_markers and (1 not in marker_set) and (6 not in marker_set):
                print(f"[DEBUG] Cleaning orphaned preview markers at ({r}, {c}) because no bomb (1) and no explosion (6) are present here.")
                self._wipe_preview_markers(r, c)
    
    def _trigger_immediate_explosion(self, bomb_r, bomb_c, blast_range=2):
        """Immediately trigger an explosion at the given position.
        This function follows the preview markers instead of recalculating blast path.
        
        Args:
            bomb_r (int): Row of the bomb to explode
            bomb_c (int): Column of the bomb to explode
            blast_range (int): Blast radius for the explosion (not used - follows preview)
        """
        print(f"[DEBUG] Triggering immediate explosion at ({bomb_r}, {bomb_c})")
        
        # First check all positions in the game for preview markers (2,3,4,5)
        # and convert them to explosions if they belong to this bomb
        for r in self.movable_objects.keys():
            for c in list(self.movable_objects[r].keys()):
                # Get the set of markers at this position
                value_set = self.movable_objects[r][c]
                
                # If this tile has any preview marker (2,3,4,5), it's in the blast radius
                if any(marker in value_set for marker in [2, 3, 4, 5]):
                    # Clean the old preview markers
                    self._wipe_preview_markers(r, c)
                    # Add explosion
                    self.movable_objects[r][c].add(6)
                    # If there's a bomb, remove it
                    if 1 in self.movable_objects[r][c]:
                        self.movable_objects[r][c].remove(1)
                        self._bomb_dict[r][c] = 0
                    # Destroy wall if present
                    if self.board[r, c] == 2:
                        print(f"[DEBUG] BLOCK DESTROYED at position ({r}, {c}) by explosion from ({bomb_r}, {bomb_c})")
                        self.board[r, c] = 1
        
        # Ensure the bomb position itself explodes
        self._wipe_preview_markers(bomb_r, bomb_c)
        if 1 in self.movable_objects[bomb_r][bomb_c]:
            self.movable_objects[bomb_r][bomb_c].remove(1)
        self.movable_objects[bomb_r][bomb_c].add(6)
        self.board[bomb_r, bomb_c] = 1
    
    def _bomb(self, pos_r, pos_c, player_id, blast_range=None):
        if blast_range is None:
            blast_range = self.game_rules.get("bomb_blast_radius", 2)

        # Create and add BombObject to track ownership and details
        # The timer in _bomb_dict starts at 1 and explodes when it reaches 5 (4 ticks after placement)
        # So, initial timer for BombObject can be 4.
        bomb_obj = BombObject(r=pos_r, c=pos_c, timer=4, player_id=player_id, blast_radius=blast_range)
        self.bombs.append(bomb_obj)
        
        self.movable_objects[pos_r][pos_c].update([1]) # Marker for bomb presence

        # make new bomb tracker for explosion logic (timer starts at 1, explodes at 5)
        self._bomb_dict[pos_r][pos_c] = 1

        # add blast range previews
        advance_r_plus = True
        advance_r_min = True
        advance_c_plus = True
        advance_c_min = True
        for b in range(blast_range + 1):
            # DOWN
            if (pos_r + b) in range(self.board.shape[0]) and advance_r_plus:
                # always mark a blast on this tile
                self.movable_objects[pos_r + b][pos_c].update([2])
                # if it's a destructible block, we'll destroy it next update and stop further spread
                if self.board[pos_r + b, pos_c] == 2:
                    advance_r_plus = False
            
            # UP
            if (pos_r - b) in range(self.board.shape[0]) and advance_r_min:
                self.movable_objects[pos_r - b][pos_c].update([2])
                if self.board[pos_r - b, pos_c] == 2:
                    advance_r_min = False
            
            # RIGHT
            if (pos_c + b) in range(self.board.shape[1]) and advance_c_plus:
                self.movable_objects[pos_r][pos_c + b].update([2])
                if self.board[pos_r, pos_c + b] == 2:
                    advance_c_plus = False
            
            # LEFT
            if (pos_c - b) in range(self.board.shape[1]) and advance_c_min:
                self.movable_objects[pos_r][pos_c - b].update([2])
                if self.board[pos_r, pos_c - b] == 2:
                    advance_c_min = False

    def _update_blast(self):
        for r in self.movable_objects.keys():
            # Create a list snapshot of keys to avoid mutation issues during iteration
            for c in list(self.movable_objects[r].keys()):
                value_set = self.movable_objects[r][c]
                # advance / remove bombs
                if self._bomb_dict[r][c] in [1, 2, 3, 4]:  # Now bombs tick from 1 to 5
                    # advance bomb one level
                    self._bomb_dict[r][c] += 1
                if self._bomb_dict[r][c] == 5:  # Explode at stage 5 instead of 4
                    # Clear preview markers for THIS bomb before chain reaction check
                    self._clear_specific_bomb_previews(r, c)

                    # Mark the BombObject as exploded to stop rendering and prepare for removal
                    for bomb_obj in self.bombs:
                        if bomb_obj.position == (r, c) and not bomb_obj.exploded:
                            bomb_obj.exploded = True
                            break # Found the bomb that just timed out
                    
                    # bomb exploded; update game state
                    self._bomb_dict[r][c] = 0
                    if 1 in self.movable_objects[r][c]: # Safety check for bomb marker
                        self.movable_objects[r][c].remove(1)
                    
                    # Promote to explosion marker
                    self.movable_objects[r][c].add(6)  # Add explosion marker
                    self.board[r, c] = 1  # Destroy block under the bomb
                    
                    # Check for chain reactions - this bomb might trigger other bombs
                    self._check_bomb_chain_reaction(r, c)

                # update blast (using original order from original code)
                if 5 in value_set:  # Was stage 4, now stage 5
                    self.movable_objects[r][c].remove(5)
                    self.movable_objects[r][c].update([6])  # Now using stage 6 for explosion
                if 4 in value_set:  # Was stage 3, now stage 4
                    self.movable_objects[r][c].remove(4)
                    self.movable_objects[r][c].update([5])
                if 3 in value_set:  # Was stage 2, now stage 3
                    self.movable_objects[r][c].remove(3)
                    self.movable_objects[r][c].update([4])
                if 2 in value_set:  # Still stage 2 for initial placement
                    self.movable_objects[r][c].remove(2)
                    self.movable_objects[r][c].update([3])
                    
                # turn blocks into ground when hit by explosions
                if 6 in self.movable_objects[r][c]:  # Was stage 5, now stage 6
                    self.board[r, c] = 1
                    # DO NOT remove stage 6 yet, leave it for player death checks

    def check_players_status(self):
        player_status = dict()
        for player in self.players:
            player_r, player_c = player.position
            # if bomb detonation affects the player position
            if 6 in self.movable_objects[player_r][player_c]:  # Now checking for stage 6 explosion
                player.alive = False
            player_status[player] = player.alive
        return player_status

    def check_game_status(self):
        # if the amount of active players is less or equal than 1
        if list(self.check_players_status().values()).count(True) <= 1:
            self.ended = True
            if self.verbose:
                print('GAME OVER')
        return

    def update_frame(self):
        if self.ended:
            print('The game has already ended.')
            return False
        assert self.frame > 0, 'Start the game first'
        assert None not in [value for key, value in
                            self.player_action_queue.items()], 'Not all player actions have been defined yet'


        # update frame
        self.frame += 1

        # update blast radius
        self._update_blast()

        # update the player positions on the board
        for player_name, action in self.player_action_queue.items():
            # get player instance
            player = self.player_name_to_object[player_name]

            if isinstance(action, tuple):
                # move player
                new_position_r, new_position_c = action
                self._move_player(new_position_r, new_position_c, player)
            elif isinstance(action, str):
                if action == 'bomb':
                    # drop bomb
                    self._bomb(player.position[0], player.position[1], player.id)
            # empty action queue
            self.player_action_queue[player_name] = None

        # First check if any players died in explosions
        self.check_players_status()
        
        # Remove all stage 6 explosions after player deaths are checked
        for r_key in list(self.movable_objects.keys()): # Iterate over a copy of keys for rows
            for c_key in list(self.movable_objects[r_key].keys()): # Iterate over a copy of keys for columns
                if 6 in self.movable_objects[r_key][c_key]:
                    self.movable_objects[r_key][c_key].remove(6)

        # After explosions are cleared, clean any orphaned preview markers from bombs that just exploded
        self._clean_all_orphaned_preview_markers()

        # Remove BombObjects that have been marked as exploded from the active list
        # This ensures they are not considered for preview refreshing or subsequent logic
        self.bombs = [b for b in self.bombs if not b.exploded]

        # Then, refresh previews for all bombs that are still active and ticking
        self._refresh_all_active_bomb_previews()
        
        # Then check if game ends (Player 1 win, Player 2 win, Draw)
        self.check_game_status()
        return True

    def get_status_dict(self):
        d = {}

        ## GAME PROPERTIES ##
        d['game_properties'] = {}

        # add game id
        d['game_properties']['id'] = self.id

        # add board dimensions
        d['game_properties']['board_dimensions'] = self.board.shape

        # add game status
        if self.ended:
            draw = True
            for player in self.players:
                if player.alive:
                    d['game_properties']['outcome'] = 'player_{}'.format(-player.id)
                    draw = False
            if draw:
                d['game_properties']['outcome'] = 'draw'
        else:
            d['game_properties']['outcome'] = 'ongoing'

        # add frame
        d['game_properties']['frame'] = self.frame

        ## BOARD POSITIONS ##
        d['board_positions'] = {}

        # add player positions
        d['board_positions']['players'] = {}
        for player in self.players:
            d['board_positions']['players']['player_{}'.format(-player.id)] = player.position

        # add void positions
        void_pos = np.where(self.board == 0)
        d['board_positions']['void'] = [(void_pos[0][idx], void_pos[1][idx]) for idx in range(len(void_pos[0]))]

        # add land positions
        land_pos = np.where(self.board == 1)
        d['board_positions']['land'] = [(land_pos[0][idx], land_pos[1][idx]) for idx in range(len(land_pos[0]))]

        # add block positions
        block_pos = np.where(self.board == 2)
        d['board_positions']['block'] = [(block_pos[0][idx], block_pos[1][idx]) for idx in range(len(block_pos[0]))]

        # add bomb positions and set stage
        bomb_pos = []
        for r in self._bomb_dict.keys():
            for c in self._bomb_dict[r].keys():
                stage = self._bomb_dict[r][c]
                # if there is a bomb, save the stage
                if stage >= 1:
                    bomb_pos.append((r, c, stage))
        d['board_positions']['bombs_and_stage'] = bomb_pos

        # add blast radius and stage
        for blast_stage in [2, 3, 4, 5, 6]:  # Added stage 6 for the new explosion stage
            # create field
            field = 'blast_radius_{}'.format(blast_stage - 1)
            d['board_positions'][field] = []

            # check where to find this blast stage
            for r in self.movable_objects.keys():
                for c in self.movable_objects[r].keys():
                    # if this particular blast stage is found
                    if blast_stage in self.movable_objects[r][c]:
                        d['board_positions'][field].append((r, c))

        return d


def validate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        player = args[0]
        assert player.game.frame > 0, 'You must start the game first! Use game.start().'
        func(*args, **kwargs)
    return wrapper


class Player:
    def __init__(self, game_instance, player_name):
        self.game = game_instance
        self.name = player_name
        self.id = None
        self.alive = True
        self.position = (None, None)
        self.power = 1 # Default bomb power (blast radius)

        # add player to the game
        self.game._add_player(self)

        # move history
        self.history = []

    @validate
    def Up(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r - 1, current_location_c

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('up')

    @validate
    def Down(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r + 1, current_location_c

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('down')

    @validate
    def Left(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r, current_location_c - 1

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('left')

    @validate
    def Right(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r, current_location_c + 1

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('right')

    @validate
    def Still(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r, current_location_c

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('still')

    @validate
    def Bomb(self):
        # add next to queue
        self.game.player_action_queue[self.name] = 'bomb'

        # save move
        self.history.append('bomb')