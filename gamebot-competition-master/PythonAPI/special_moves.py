"""
Special move executor for Street Fighter 2 AI bot.
Executes predefined move sequences when triggered by game state conditions.
"""
import random

class SpecialMoveExecutor:
    """Handles special move execution with proper timing."""
    
    def __init__(self):
        self.move_timer = 0
        self.current_sequence = None
        self.sequence_step = 0
        
        # Define special move sequences (button combinations per frame)
        self.moves = {
            'fireball_lp': [
                {'Down': True},                              # Frame 1: Hold down
                {'Down': True, 'Right': True},               # Frame 2: Down-forward
                {'Right': True, 'A': True},                  # Frame 3: Forward + LP
                {'Right': True, 'A': True},                  # Frame 4: Hold
                {},                                          # Frame 5: Release
            ],
            'fireball_hp': [
                {'Down': True},
                {'Down': True, 'Right': True},
                {'Right': True, 'Y': True},                  # HP instead of LP
                {'Right': True, 'Y': True},
                {},
            ],
            'dragon_punch_lp': [
                {'Right': True},                             # Frame 1: Forward
                {'Down': True},                              # Frame 2: Down
                {'Down': True, 'Right': True, 'A': True},   # Frame 3: Down-forward + LP
                {'Down': True, 'Right': True, 'A': True},   # Frame 4: Hold
                {},                                          # Frame 5: Release
            ],
            'dragon_punch_hp': [
                {'Right': True},
                {'Down': True},
                {'Down': True, 'Right': True, 'Y': True},
                {'Down': True, 'Right': True, 'Y': True},
                {},
            ],
            'hurricane_lk': [
                {'Down': True},                              # Frame 1: Down
                {'Down': True},                              # Frame 2: Down again
                {'Up': True, 'L': True},                     # Frame 3: Up + LK
                {'Up': True, 'L': True},                     # Frame 4: Hold
                {},                                          # Frame 5: Release
            ],
            'hurricane_hk': [
                {'Down': True},
                {'Down': True},
                {'Up': True, 'R': True},                     # HK instead of LK
                {'Up': True, 'R': True},
                {},
            ],
        }
    
    def should_execute_special(self, game_state, opponent_distance='close'):
        """Decide whether to execute a special move based on game state."""
        # Don't interrupt if already in a sequence
        if self.current_sequence:
            return False
        
        p1 = game_state.player1
        p2 = game_state.player2
        
        # Check conditions for special moves
        can_attack = not p1.is_player_in_move
        has_health = p1.health > 0
        
        if not can_attack or not has_health:
            return False
        
        # Calculate distance to opponent
        distance = abs(p1.x_coord - p2.x_coord)
        
        # Adjust special move chance based on distance
        # Closer = higher chance for dragon punch, farther = fireball
        if distance < 100:  # Very close
            special_chance = 0.25  # 25% chance
        elif distance < 200:  # Medium range
            special_chance = 0.20  # 20% chance
        else:  # Far away
            special_chance = 0.15  # 15% chance
        
        # Random chance to try special move
        if random.random() < special_chance:
            return True
        
        return False
    
    def get_next_frame(self):
        """Get the button presses for the current step in the sequence."""
        if not self.current_sequence:
            return {}
        
        if self.sequence_step >= len(self.current_sequence):
            # Sequence complete
            buttons = {}
            self.current_sequence = None
            self.sequence_step = 0
            return buttons
        
        buttons = self.current_sequence[self.sequence_step].copy()
        self.sequence_step += 1
        
        return buttons
    
    def start_move(self, move_name=None, game_state=None):
        """Start executing a special move sequence."""
        if move_name is None and game_state:
            # Choose move based on opponent distance
            p1 = game_state.player1
            p2 = game_state.player2
            distance = abs(p1.x_coord - p2.x_coord)
            
            if distance < 100:  # Close range - dragon punch or hurricane
                move_name = random.choice(['dragon_punch_lp', 'dragon_punch_hp', 'hurricane_lk'])
            elif distance < 200:  # Medium range - fireball or poke
                move_name = random.choice(['fireball_lp', 'fireball_hp', 'dragon_punch_lp'])
            else:  # Far away - fireball to close gap
                move_name = random.choice(['fireball_lp', 'fireball_hp'])
        elif move_name is None:
            # Pick random move
            move_name = random.choice(list(self.moves.keys()))
        
        if move_name in self.moves:
            self.current_sequence = self.moves[move_name]
            self.sequence_step = 0
            return True
        return False
    
    def reset(self):
        """Reset the executor (e.g., between rounds)."""
        self.current_sequence = None
        self.sequence_step = 0
        self.move_timer = 0
