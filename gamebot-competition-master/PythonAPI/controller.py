import socket
import json
import sys
import time
from game_state import GameState
from command import Command
from buttons import Buttons
from listen_to_key import get_current_keypress
from make_dataset import record_frame

player_id = sys.argv[1]
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else 'bot'
MODEL_TYPE = 'rnn' if len(sys.argv) > 2 and sys.argv[2] == 'rnn' else 'standard'
port = 9999 if player_id == '1' else 10000

def connect(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    client, _ = s.accept()
    print('Connected to game!')
    return client

def send(sock, cmd):
    payload = json.dumps(cmd.object_to_dict())
    print("[Controller] Sending:", payload)
    print("\n[Controller] Detailed Debug:")
    print("1. Command object button states:", vars(cmd.player_buttons))
    print("2. Serialized command:", payload)
    print("3. Socket info:", sock.getsockname(), "->", sock.getpeername())
    sock.sendall(payload.encode())

def receive(sock):
    data = json.loads(sock.recv(4096).decode())
    return GameState(data)

def main():
    sock = connect(port)
    cmd = Command()
    player_id_set = False
    while True:
        gs = receive(sock)
    
        if MODE != 'record' and not player_id_set:
            from bot import Bot
            print("[Controller] Using standard model bot")
                
            bot = Bot(player_id=gs.player1.player_id)
            player_id_set = True

        if MODE == 'record':
            keys = get_current_keypress()
            # Forward human input
            cmd.player_buttons = Buttons({k: True for k in keys})
            record_frame(gs, keys)
        else:
            print("\n[Controller] Getting bot command...")
            cmd = bot.fight(gs, player_id)
            print(f"[Controller] Bot command received: {cmd.object_to_dict()}")
        
        send(sock, cmd)
        time.sleep(1/60.0)
        
if __name__ == '__main__':
    main()