import os
import csv
import time
import joblib
import tensorflow as tf
from collections import deque
from datetime import datetime
from command import Command
from buttons import Buttons
from device_utils import get_device
from special_moves import SpecialMoveExecutor

# Per-frame console printing is expensive (each print() is effectively a
# flush/syscall) and this file was doing 10+ of them every single frame.
# Off by default - set SF2_DEBUG=1 to see prediction probabilities per frame.
DEBUG = os.environ.get('SF2_DEBUG', '0') == '1'

# define constants the same way as done when training
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over','player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id','player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id','diff_x', 'diff_y', 'diff_health']
# making feature columns just like during training time - order must stay
# in lockstep with how the scaler was originally fit
FEATURE_COLS = []
for t in range(WINDOW_SIZE-1, -1, -1):
    suffix = f"_t-{t}"
    for feat in STATE_FEATURES:
        FEATURE_COLS.append(feat + suffix)
BUTTONS = ['Up', 'Down', 'Right', 'Left', 'Y', 'B', 'X', 'A', 'L', 'R']
P1_BUTTON_COLS = [f'player1_buttons_{b.lower()}' for b in BUTTONS]

FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}


class Bot:
    def __init__(self,player_id=0, model_path=None):
        self.buttons = Buttons()
        self.cmd = Command()
        
        # Match logging (Required by Competition Rule #9)
        self.match_log = []
        self.frame_count = 0
        self.character_id = None
        self.player_id_str = str(player_id)
        self.log_start_time = time.time()
        
        # Pattern breaking - track recent actions to avoid repetition
        self.recent_actions = []  # Last 10 button combinations
        self.max_history = 10
        self.no_repeat_frames = 5  # Don't repeat exact same action for N frames
        
        # Special move executor
        self.special_executor = SpecialMoveExecutor()
        
        # locate model & scaler
        if model_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            model_path = os.path.join(base, 'model_'+f'{player_id}' +'.keras')

        # ---- auto-detect GPU, fall back to CPU automatically ----
        self.device = get_device()

        with tf.device(self.device):
            self.model = tf.keras.models.load_model(model_path)

        scaler = joblib.load(model_path + '.scaler')
        # Move the scaler's mean/scale into the graph as constants so scaling
        # happens on-device inside one compiled call instead of building a
        # fresh pandas DataFrame + running sklearn's transform() every frame.
        self.scaler_mean_ = tf.constant(scaler.mean_, dtype=tf.float32)
        self.scaler_scale_ = tf.constant(scaler.scale_, dtype=tf.float32)

        n_features = len(FEATURE_COLS)

        def _make_compiled_predict(use_xla):
            @tf.function(
                input_signature=[tf.TensorSpec(shape=[1, n_features], dtype=tf.float32)],
                jit_compile=use_xla,
            )
            def compiled_predict(x):
                x_scaled = (x - self.scaler_mean_) / self.scaler_scale_
                return self.model(x_scaled, training=False)
            return compiled_predict

        with tf.device(self.device):
            dummy = tf.zeros((1, n_features), dtype=tf.float32)
            try:
                # Try XLA first - fastest option when the model's ops support it.
                self._compiled_predict = _make_compiled_predict(use_xla=True)
                for _ in range(3):
                    _ = self._compiled_predict(dummy)
            except Exception as e:
                # Some layer/op in the model may not be XLA-compatible - fall
                # back to a plain compiled graph rather than crashing the bot.
                print(f"[Bot] XLA compilation failed ({e}); falling back to standard @tf.function.")
                self._compiled_predict = _make_compiled_predict(use_xla=False)
                for _ in range(3):
                    _ = self._compiled_predict(dummy)

        print(f"[Bot] Model loaded and warmed up on {self.device}.")

        # init frame buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)
        empty = {feat: 0 for feat in STATE_FEATURES}
        empty['fight_result'] = 'NOT_OVER'
        for _ in range(WINDOW_SIZE):
            self.buffer.append(empty.copy())

    def _frame_to_dict(self, gs):
        # map GameState to raw feature dict (no suffix)
        p1, p2 = gs.player1, gs.player2
        d = {
            'timer': gs.timer,
            'fight_result': gs.fight_result,
            'has_round_started': int(gs.has_round_started),
            'is_round_over': int(gs.is_round_over),
            'player1_id': p1.player_id,
            'p1_health': p1.health,
            'p1_x': p1.x_coord,
            'p1_y': p1.y_coord,
            'p1_jumping': int(p1.is_jumping),
            'p1_crouching': int(p1.is_crouching),
            'p1_in_move': int(p1.is_player_in_move),
            'p1_move_id': p1.move_id,
            'player2_id': p2.player_id,
            'p2_health': p2.health,
            'p2_x': p2.x_coord,
            'p2_y': p2.y_coord,
            'p2_jumping': int(p2.is_jumping),
            'p2_crouching': int(p2.is_crouching),
            'p2_in_move': int(p2.is_player_in_move),
            'p2_move_id': p2.move_id,
            'diff_x': p1.x_coord - p2.x_coord,
            'diff_y': p1.y_coord - p2.y_coord,
            'diff_health': p1.health - p2.health,
        }
        return d

    def fight(self, gs, player_id):
        # Track frame count and character ID for logging
        self.frame_count += 1
        if self.character_id is None:
            self.character_id = gs.player1.player_id
        
        # 1. append new frame
        raw = self._frame_to_dict(gs)
        self.buffer.append(raw)

        # 2. build flattened feature list (same column order as training)
        flat = []
        for t in range(WINDOW_SIZE-1, -1, -1):
            frame = self.buffer[WINDOW_SIZE-1 - t]
            for feat in STATE_FEATURES:
                val = frame[feat]
                flat.append(FIGHT_MAP[val] if feat == 'fight_result' else int(val))

        # 3. scale + predict in one compiled, on-device call
        #    (replaces the old per-frame pandas DataFrame + scaler.transform()
        #    + model.predict() - model.predict() in particular carries large
        #    fixed overhead per call that dwarfs actual compute for a single
        #    sample; calling the compiled function directly avoids all of it)
        x = tf.constant([flat], dtype=tf.float32)
        preds = self._compiled_predict(x).numpy()[0]

        if DEBUG:
            print("\nPrediction probabilities for each button:")
            for button, prob in zip(BUTTONS, preds):
                if prob > 0.005:
                    print(f"{button}: {prob:.2%}")

        # 4. map to Buttons, resolving opposing directions
        probs = {b: float(preds[i]) for i, b in enumerate(BUTTONS)}
        
        # Determine if we need to mirror directional output based on which side of the screen the bot is on
        # Training data always had self as player1, usually on left side facing right
        # When bot is on right side (x > opponent_x), forward is Left instead of Right
        p1 = gs.player1
        p2 = gs.player2
        
        # Figure out which player slot corresponds to the bot
        if player_id == "1":
            bot_x = p1.x_coord
            opp_x = p2.x_coord
        else:
            bot_x = p2.x_coord
            opp_x = p1.x_coord
        
        # Mirror needed when bot is on the right side of the screen
        needs_mirror = bot_x > opp_x
        
        # Use different thresholds: lower for directional movement, higher for attacks
        # This ensures we capture movement even when model is uncertain
        directional_threshold = 0.05  # Lower threshold for Up/Down/Left/Right
        attack_threshold = 0.10       # Higher threshold for buttons
        
        btn_map = {}
        for b in BUTTONS:
            if b in ['Up', 'Down', 'Left', 'Right']:
                btn_map[b] = (probs[b] > directional_threshold)
            else:
                btn_map[b] = (probs[b] > attack_threshold)
        
        # If bot is on right side, swap Left and Right predictions
        if needs_mirror:
            btn_map['Left'], btn_map['Right'] = btn_map['Right'], btn_map['Left']
            probs['Left'], probs['Right'] = probs['Right'], probs['Left']
        
        # Resolve opposing directions - only allow one direction per axis
        if btn_map['Left'] and btn_map['Right']:
            if probs['Left'] > probs['Right']:
                btn_map['Right'] = False
            else:
                btn_map['Left'] = False
        
        if btn_map['Up'] and btn_map['Down']:
            if probs['Up'] > probs['Down']:
                btn_map['Down'] = False
            else:
                btn_map['Up'] = False
        
        # Limit total buttons to prevent mashing - keep only top predictions
        active_count = sum(1 for v in btn_map.values() if v)
        if active_count > 3:
            # Sort by probability and keep only top 3, but always preserve at least one directional input
            sorted_btns = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            # Find which directional buttons are active
            active_directionals = [b for b in ['Up', 'Down', 'Left', 'Right'] if btn_map[b]]
            
            # Keep top 3 buttons
            top_buttons = {b for b, p in sorted_btns[:3] if p > (0.05 if b in ['Up', 'Down', 'Left', 'Right'] else 0.10)}
            
            # If we lost all directional inputs in filtering, add back the best one
            has_directional_in_top = any(b in top_buttons for b in ['Up', 'Down', 'Left', 'Right'])
            if not has_directional_in_top and active_directionals:
                # Add the directional button with highest probability
                best_dir = max(active_directionals, key=lambda b: probs[b])
                top_buttons.add(best_dir)
                # Remove lowest attack button to make room
                if len(top_buttons) > 3:
                    attack_btns_in_top = [b for b in top_buttons if b not in ['Up', 'Down', 'Left', 'Right']]
                    if attack_btns_in_top:
                        worst_attack = min(attack_btns_in_top, key=lambda b: probs[b])
                        top_buttons.discard(worst_attack)
            
            btn_map = {b: (b in top_buttons) for b in BUTTONS}
            
            # Re-resolve directions after filtering
            if btn_map['Left'] and btn_map['Right']:
                if probs['Left'] > probs['Right']:
                    btn_map['Right'] = False
                else:
                    btn_map['Left'] = False
            
            if btn_map['Up'] and btn_map['Down']:
                if probs['Up'] > probs['Down']:
                    btn_map['Down'] = False
                else:
                    btn_map['Up'] = False
        
        # Pattern breaking: avoid repeating exact same action too long
        current_action = tuple(sorted([b for b, v in btn_map.items() if v]))
        
        # Check if we've been doing the same thing for too many frames
        if len(self.recent_actions) >= self.no_repeat_frames:
            recent_unique = set(self.recent_actions[-self.no_repeat_frames:])
            if len(recent_unique) == 1 and recent_unique.pop() == current_action and current_action:
                # Force variety - pick second-best prediction or inject randomness
                import random
                alternative_btns = [b for b, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[1:4] if p > 0.05]
                if alternative_btns:
                    # Clear current action and try something different
                    for b in BUTTONS:
                        btn_map[b] = False
                    # Pick a random alternative
                    chosen = random.choice(alternative_btns)
                    btn_map[chosen] = True
                    
                    # If it's a directional button, sometimes add an attack
                    if chosen in ['Up', 'Down', 'Left', 'Right']:
                        attack_btns = [b for b in ['A', 'L', 'Y'] if probs[b] > 0.05]
                        if attack_btns and random.random() > 0.5:
                            btn_map[random.choice(attack_btns)] = True
        
        # Forced movement injection: ensure bot moves directionally every few frames
        has_directional = any(btn_map[d] for d in ['Up', 'Down', 'Left', 'Right'])
        no_movement_frames = 8  # Force movement if none detected for N frames
        
        if not has_directional and len(self.recent_actions) >= no_movement_frames:
            recent_has_movement = False
            for action in self.recent_actions[-no_movement_frames:]:
                if any(b in action for b in ['Up', 'Down', 'Left', 'Right']):
                    recent_has_movement = True
                    break
            
            if not recent_has_movement:
                # Force directional movement based on opponent distance
                import random
                diff_x = gs.player1.x_coord - gs.player2.x_coord
                
                # Decide direction: move toward opponent 60% of time, away 40%
                if abs(diff_x) < 100:  # Close range
                    # Move away to create spacing
                    forced_dir = 'Left' if diff_x > 0 else 'Right'
                else:
                    # Move toward opponent
                    forced_dir = 'Right' if diff_x > 0 else 'Left'
                
                # Sometimes jump instead of walk (30% chance)
                if random.random() < 0.3:
                    btn_map['Up'] = True
                else:
                    btn_map[forced_dir] = True
                
                # Add occasional attack with movement (40% chance)
                if random.random() < 0.4:
                    attack_choices = [b for b in ['A', 'L', 'Y'] if probs[b] > 0.08]
                    if attack_choices:
                        btn_map[random.choice(attack_choices)] = True
        
        # Track this action
        self.recent_actions.append(current_action)
        if len(self.recent_actions) > self.max_history:
            self.recent_actions.pop(0)
        
        # Special move execution - override model predictions when appropriate
        if self.special_executor.current_sequence:
            # Continue current special move sequence
            special_buttons = self.special_executor.get_next_frame()
            if special_buttons:
                # Override button map with special move inputs
                for b in BUTTONS:
                    btn_map[b] = False  # Clear all buttons
                for btn, pressed in special_buttons.items():
                    if btn in btn_map:
                        btn_map[btn] = pressed
                # Mirror directional inputs if bot is on right side
                if needs_mirror and ('Left' in btn_map or 'Right' in btn_map):
                    btn_map['Left'], btn_map['Right'] = btn_map['Right'], btn_map['Left']
        elif self.special_executor.should_execute_special(gs):
            # Start a new special move (choose based on distance)
            self.special_executor.start_move(game_state=gs)
            special_buttons = self.special_executor.get_next_frame()
            if special_buttons:
                for b in BUTTONS:
                    btn_map[b] = False
                for btn, pressed in special_buttons.items():
                    if btn in btn_map:
                        btn_map[btn] = pressed
                # Mirror directional inputs if bot is on right side
                if needs_mirror and ('Left' in btn_map or 'Right' in btn_map):
                    btn_map['Left'], btn_map['Right'] = btn_map['Right'], btn_map['Left']

        cmd = Command()
        if player_id == "1":
            cmd.player_buttons = Buttons(btn_map)
        else:
            cmd.player2_buttons = Buttons(btn_map)

        if DEBUG:
            active_buttons = [btn for btn, state in btn_map.items() if state]
            print(f"\nActive buttons for Player {player_id}:", ", ".join(active_buttons) if active_buttons else "None")
        
        # Log this frame to CSV (Required by Competition Rule #9)
        elapsed_time = time.time() - self.log_start_time
        buttons_pressed = '+'.join(b for b in BUTTONS if btn_map[b])
        
        self.match_log.append({
            'frame': self.frame_count,
            'time_seconds': f"{elapsed_time:.2f}",
            'player_side': f'P{player_id}',
            'character_id': self.character_id,
            'buttons_pressed': buttons_pressed if buttons_pressed else 'NONE',
            'p1_health': gs.player1.health,
            'p2_health': gs.player2.health,
            'timer': gs.timer
        })
        
        # Save log every 100 frames to avoid excessive I/O
        if self.frame_count % 100 == 0:
            self._save_match_log(player_id)

        return cmd
    
    def _save_match_log(self, player_id):
        """Save match log to CSV file (Required by Competition Rule #9)"""
        if not self.match_log:
            return
        
        import pandas as pd
        df = pd.DataFrame(self.match_log)
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), '..', 'match_logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate filename based on stage and fight number
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stage1_fight_P{player_id}_{timestamp}.csv"
        filepath = os.path.join(logs_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"[Bot] Match log saved: {filepath} ({len(df)} frames)")