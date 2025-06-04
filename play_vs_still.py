import argparse
import cv2
import time
import numpy as np
from game import Game, Player
from render_tool import RenderTool, MapScheme
import copy

def main():
    # Choose the map
    map_scheme = MapScheme().standard
    
    # Initialize the game
    game = Game(map_scheme)
    render_tool = RenderTool(game)
    
    # Create players - you control player1, player2 is still
    player1 = Player(game, 'player1')  # Human player
    player2 = Player(game, 'player2')  # Still opponent
    
    # Start the game
    game.start()
    img = render_tool.render_current_frame(save_media=True)
    
    img = cv2.imread(render_tool.image_path + '{}.png'.format(game.frame))
    cv2.namedWindow(game.id)
    cv2.imshow(game.id, img)
    
    print("\nPlaying against a still opponent")
    print("Controls:")
    print("W - Move Up")
    print("S - Move Down")
    print("A - Move Left")
    print("D - Move Right")
    print("SPACE - Place Bomb")
    print("ESC - Quit Game\n")
    
    # Track score and game state
    blocks_destroyed = 0
    initial_blocks = count_blocks(game)
    
    while True:
        # Get key press for human player (player1)
        k = cv2.waitKey(0)
        if k == 27:  # ESCAPE KEY
            print('Escape key pressed. Closing game..')
            break
        
        old_pos = player1.position
        print(f"[DEBUG] Before action - Player position: {old_pos}")
        print(f"[DEBUG] Movable objects at current pos: {game.movable_objects[old_pos[0]][old_pos[1]]}")
        print(f"[DEBUG] Bomb dict at current pos: {game._bomb_dict[old_pos[0]][old_pos[1]]}")
        
        # Process human player's action
        if k == 119:  # UP (W key)
            print("You chose: UP")
            player1.Up()
        elif k == 115:  # DOWN (S Key)
            print("You chose: DOWN")
            player1.Down()
        elif k == 97:  # LEFT (A key)
            print("You chose: LEFT")
            player1.Left()
        elif k == 100:  # RIGHT (D key)
            print("You chose: RIGHT")
            player1.Right()
        elif k == 32:  # BOMB (SPACE BAR)
            print("You chose: BOMB")
            player1.Bomb()
        else:
            print(f"Not recognized key {k} pressed. Standing still for this action..")
            player1.Still()
        
        # Still player does nothing
        player2.Still()
        
        try:
            # Debug before update
            print(f"[DEBUG] Before update_frame - Player action queue: {game.player_action_queue}")
            
            # Make a snapshot of the current state for comparison
            old_board = copy.deepcopy(game.board)
            old_movable = copy.deepcopy(game.movable_objects)
            old_bombs = copy.deepcopy(game._bomb_dict)
            
            # Update game state
            game.update_frame()
            
            # Debug after update
            new_pos = player1.position
            print(f"[DEBUG] After update_frame - Player position: {new_pos}")
            if new_pos[0] >= 0 and new_pos[1] >= 0:  # Make sure position is valid
                print(f"[DEBUG] Movable objects at new pos: {game.movable_objects[new_pos[0]][new_pos[1]]}")
                print(f"[DEBUG] Bomb dict at new pos: {game._bomb_dict[new_pos[0]][new_pos[1]]}")
                
                # Check for bombs and blast radiuses in adjacent tiles
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = new_pos[0] + dr, new_pos[1] + dc
                    if 0 <= nr < game.board.shape[0] and 0 <= nc < game.board.shape[1]:
                        movable = game.movable_objects[nr][nc]
                        bomb_val = game._bomb_dict[nr][nc]
                        print(f"[DEBUG] Adjacent tile ({nr},{nc}): Objects={movable}, Bomb={bomb_val}")
            
            # Debug player alive status
            print(f"[DEBUG] Player1 alive: {player1.alive}, Player2 alive: {player2.alive}")
            
            # Enhanced visualization of bombs and explosions
            explosion_map = np.zeros_like(game.board)
            bomb_map = np.zeros_like(game.board)
            
            # Mark bombs and explosions on the map
            for r in range(game.board.shape[0]):
                for c in range(game.board.shape[1]):
                    if 5 in game.movable_objects[r][c]:
                        explosion_map[r, c] = 2  # Active explosion
                    elif any(blast in game.movable_objects[r][c] for blast in [2, 3, 4]):
                        explosion_map[r, c] = 1  # Future blast area
                    
                    if 1 in game.movable_objects[r][c]:
                        bomb_timer = game._bomb_dict[r][c]
                        if bomb_timer > 0:
                            bomb_map[r, c] = bomb_timer
            
            if np.any(explosion_map) or np.any(bomb_map):
                print("\n--- DANGER MAP ---")
                print("Active explosions (2) - YOU WILL DIE IF STANDING HERE")
                print("Future blasts (1) - Explosion coming soon")
                print(explosion_map)
                print("\n--- BOMB MAP ---")
                print("Bomb timers (1-4): Higher number = closer to explosion")
                print(bomb_map)
            
            render_tool.render_current_frame(save_media=True)
            img = cv2.imread(render_tool.image_path + '{}.png'.format(game.frame))
            cv2.imshow(game.id, img)
            
            # Track blocks destroyed
            current_blocks = count_blocks(game)
            if current_blocks < initial_blocks:
                new_destroyed = initial_blocks - current_blocks
                blocks_destroyed += (new_destroyed - blocks_destroyed)
                print(f"Blocks destroyed: {blocks_destroyed}")
                initial_blocks = current_blocks
            
            # Check game end conditions
            if not player2.alive:
                print("You win! You eliminated the opponent!")
                print(f"[DEBUG] Final board state: {game.board}")
                print(f"[DEBUG] Final movable objects: {game.movable_objects}")
                time.sleep(3)
                break
            elif not player1.alive:
                print("You lost! You were hit by your own bomb!")
                print(f"[DEBUG] Final board state: {game.board}")
                print(f"[DEBUG] Final movable objects: {game.movable_objects}")
                # Check what might have killed the player
                if new_pos[0] >= 0 and new_pos[1] >= 0:  # Make sure position is valid
                    objects_at_death = game.movable_objects[new_pos[0]][new_pos[1]]
                    print(f"[DEBUG] Objects at death position: {objects_at_death}")
                    # Check if any blast numbers (2-5) are in the objects set
                    blast_numbers = [num for num in range(2, 6) if num in objects_at_death]
                    if blast_numbers:
                        print(f"[DEBUG] Player was in blast radius of objects: {blast_numbers}")
                time.sleep(3)
                break
            elif game.ended:
                print("Game ended in a draw!")
                print(f"[DEBUG] Final board state: {game.board}")
                print(f"[DEBUG] Final movable objects: {game.movable_objects}")
                time.sleep(3)
                break
                
        except Exception as exc:
            print(exc)
            import traceback
            print(traceback.format_exc())
            break
    
    # Final stats
    print(f"Final score: {blocks_destroyed} blocks destroyed")
    if not player2.alive:
        print("Victory: You eliminated the opponent!")
    cv2.destroyAllWindows()

def count_blocks(game):
    """Count the number of destructible blocks on the board"""
    count = 0
    for r in range(game.board.shape[0]):
        for c in range(game.board.shape[1]):
            if game.board[r, c] == 2:  # 2 = destructible block
                count += 1
    return count

if __name__ == "__main__":
    main()
