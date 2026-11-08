# Street Fighter II Turbo AI Competition — Participant Kit

This repository contains the Python controller, data-preprocessing scripts,
training code, and Lua integration scripts for a Street Fighter II Turbo
game-playing bot. It is intended for the participant-side single-player
workflow: record gameplay, build temporal datasets, train a character model,
and test the bot against the game CPU.

## Competition overview

Participants have 24 hours to collect gameplay data, train a model, test it against the game CPU, and submit the trained model package. After the submission deadline, organizers will run the submitted models against one another using a separate two-player tournament environment.

This participant kit contains the participant-side code. The organizer's
tournament runner is not part of this project.

## Objective

Build an AI model that observes the current Street Fighter II Turbo game state and predicts which controller buttons should be pressed next.

The supplied baseline uses imitation learning:

1. A participant plays the game.
2. The program records game states and human button presses.
3. Recorded frames are converted into six-frame training windows.
4. A neural network learns to predict button presses.
5. The trained model controls a character in the game.

Participants may improve the model and training process, provided their final submission follows the organizer's model and file-format requirements.

## Included files

```text
Street-Fighter-Participant/
├── README.md
├── bizhawk_prereqs.exe
└── gamebot-competition-master/
    ├── Controls.PNG
    ├── snes_controller_layout.png
    ├── PythonAPI/
    │   ├── bot.py
    │   ├── buttons.py
    │   ├── command.py
    │   ├── controller.py
    │   ├── device_utils.py
    │   ├── game_state.py
    │   ├── listen_to_key.py
    │   ├── make_dataset.py
    │   └── player.py
    ├── normalized_character_datasets/
    ├── flattened_window_datasets/
    ├── pre_processing/
    │   └── preprocess_windows.py
    ├── train_models/
    │   └── train_individual_character.py
    └── models/
```

Generated datasets, trained models, match logs, emulator binaries, ROMs, and
emulator state files are intentionally excluded by `.gitignore`. Obtain any
required licensed game or emulator files separately, install the single-player
environment under `gamebot-competition-master/single-player/`, and place them
at the paths expected by the scripts.

## Supported characters

| ID | Character | ID | Character |
|---:|---|---:|---|
| 0 | Ryu | 6 | Zangief |
| 1 | E. Honda | 7 | Dhalsim |
| 2 | Blanka | 8 | M. Bison |
| 3 | Guile | 9 | Sagat |
| 4 | Ken | 10 | Balrog |
| 5 | Chun-Li | 11 | Vega |

The character ID determines the dataset and model filenames:

```text
normalized_dataset_<character-id>.csv
windowed_dataset_<character-id>.csv
model_<character-id>.keras
model_<character-id>.keras.scaler
```

## Requirements

- Windows 7 or newer, 64-bit
- Python 3
- TensorFlow 2.x
- pandas
- NumPy
- scikit-learn
- joblib
- pynput

Install the Python packages with:

```powershell
pip install tensorflow pandas numpy scikit-learn joblib pynput
```

If BizHawk does not start, run `bizhawk_prereqs.exe` first.

## Game-state information

At each time step, the Python program receives:

- Round timer, start status, end status, and result
- Character IDs for both players
- Health for both players
- X and Y coordinates for both players
- Jumping and crouching states
- Whether each player is performing a move
- Move IDs
- Differences in position and health
- Current controller-button states

The baseline predicts these ten buttons in this order:

```text
UP, DOWN, RIGHT, LEFT, Y, B, X, A, L, R
```

Start and Select are not training targets.

## Step 1: Record gameplay data

1. Open `gamebot-competition-master/single-player/EmuHawk.exe`.
2. Select **File > Open ROM** and open the supplied Street Fighter II Turbo game file.
3. Select **Tools > Tool Box**, or press `Shift+T`.
4. Open a terminal in `gamebot-competition-master/PythonAPI`.
5. Start the recorder:

   ```powershell
   python controller.py 1 record
   ```

6. In the game, choose Normal mode and select a character.
7. In the Tool Box, select the second icon, **Gyroscope Bot**.
8. Wait for `Connected to game!` in the terminal.
9. Play the game using the configured keyboard controls.

The recorder creates or appends to:

```text
normalized_character_datasets/normalized_dataset_<character-id>.csv
```

The current baseline recorder saves frames in which at least one mapped gameplay key is pressed. This behavior is intentionally unchanged in the participant kit.

### Default recording controls

| Keyboard key | SNES input |
|---|---|
| Arrow keys | Up, Down, Left, Right |
| A | Y |
| S | B |
| D | X |
| Z | A |
| X | L |
| C | R |
| Enter | Start |
| Space | Select |

## Step 2: Create the windowed dataset

The recording script creates individual frame rows. The baseline training script requires six-frame temporal windows, so preprocessing is a separate required step.

From `gamebot-competition-master`, run:

```powershell
python pre_processing/preprocess_windows.py
```

The script processes all `normalized_dataset_*.csv` files it finds and creates:

```text
flattened_window_datasets/windowed_dataset_<character-id>.csv
```

The preprocessing script also removes invalid or duplicate states, removes Start and Select, and retains limited round-end information.

## Step 3: Train a model

Open:

```text
train_models/train_individual_character.py
```

Near the bottom of the file, set `characters_to_train` to the character IDs you want to train. For example:

```python
characters_to_train = [0]
```

Then run from `gamebot-competition-master`:

```powershell
python train_models/train_individual_character.py
```

The baseline model uses six frames of game-state information and predicts ten independent button probabilities. Successful training creates both:

```text
models/model_<character-id>.keras
models/model_<character-id>.keras.scaler
```

Both files are required by the supplied inference code. The scaler must correspond to the submitted model.

## Step 4: Test the trained bot

1. Confirm the model and scaler are in `gamebot-competition-master/models`.
2. Start the single-player emulator and load the game.
3. Open a terminal in `gamebot-competition-master/PythonAPI`.
4. Run:

   ```powershell
   python controller.py 1
   ```

5. Select the same character for which the model was trained.
6. Open **Gyroscope Bot** from the emulator Tool Box.

The program selects `model_<character-id>.keras` based on the chosen Player 1 character.

## Baseline model details

- Window size: 6 frames
- State features per frame: 23
- Model input size: 138 values
- Output size: 10 button probabilities
- Hidden layers: 256, 128, and 64 units
- Output activation: sigmoid
- Loss: binary cross-entropy
- Model format: Keras `.keras`
- Feature scaling: scikit-learn `StandardScaler`

Participants may modify the model architecture and training strategy, subject to the final submission rules announced by the organizers.

## Submission checklist

Unless the organizers announce a different submission format, submit:

```text
model_<character-id>.keras
model_<character-id>.keras.scaler
```

Before submitting, verify that:

- The model loads without errors.
- The scaler belongs to the same model.
- The filename contains the correct character ID.
- The model runs on the organizer-supported environment.
- The bot can complete a test round against the CPU.
- The submission does not require network access or additional private files.

Do not submit datasets, emulator files, notebooks, or the entire project unless the organizers explicitly request them.

## Competition rules to confirm with organizers

The organizers will separately announce:

- Submission deadline and upload method
- Permitted characters
- Maximum model size and inference time
- Whether external datasets or pretrained models are allowed
- Match format and number of rounds
- Crash, timeout, and invalid-submission handling
- Hardware and software versions used for the tournament

## Troubleshooting

### The emulator does not connect

- Start `controller.py` before opening Gyroscope Bot.
- Confirm no other application is using port `9999`.
- Confirm the participant controller is running as Player 1.

### The training script skips a character

Confirm that this file exists:

```text
flattened_window_datasets/windowed_dataset_<character-id>.csv
```

Also confirm that the character ID is present in `characters_to_train`.

### The bot cannot find its model

Confirm that both model files exist in `models/` and use the selected character ID.

### TensorFlow does not detect a GPU

The baseline automatically falls back to CPU. A GPU is optional. To explicitly force CPU inference in PowerShell:

```powershell
$env:SF2_FORCE_CPU = "1"
python controller.py 1
```

## Organizer note

This repository is the participant training and single-player testing kit. Official model-versus-model matches must be run using the separate organizer-controlled two-player environment.
