from pynput import keyboard

_pressed = set()

#map keybinds from keyboard to game
custom_keymap = {
    'A': 'Y',
    'S': 'B',
    'D': 'X',
    'Z': 'A',
    'X': 'L',
    'C': 'R',
    'KEY.UP': 'UP',
    'KEY.DOWN': 'DOWN',
    'KEY.LEFT': 'LEFT',
    'KEY.RIGHT': 'RIGHT',
    'KEY.ENTER': 'START',
    'KEY.SPACE': 'SELECT'
}

def on_press(key):
    try:
        key_str = key.char.upper()
        _pressed.add(key_str)
        # print(f"Normal key pressed: {key_str}")
    except AttributeError:
        #handle special keys
        key_str = f"KEY.{str(key).replace('Key.', '').upper()}"
        _pressed.add(key_str)
        # print(f"Special key pressed: {key_str}")

def on_release(key):
    try:
        key_str = key.char.upper()
        _pressed.discard(key_str)
    except AttributeError:
        key_str = f"KEY.{str(key).replace('Key.', '').upper()}"
        _pressed.discard(key_str) 

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

def get_current_keypress():
    mapped = set()
    for k in _pressed:
        if k in custom_keymap:
            mapped.add(custom_keymap[k])
        # else:
        #     print(f"Unmapped key pressed: {k}")
    return list(mapped)