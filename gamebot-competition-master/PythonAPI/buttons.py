class Buttons:
    def __init__(self, buttons_dict=None):
        if buttons_dict is not None:
            self.dict_to_object(buttons_dict)
        else:
            self.init_buttons()

    def init_buttons(self):
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.select = False
        self.start = False
        self.Y = False
        self.B = False
        self.X = False
        self.A = False
        self.L = False
        self.R = False

    def dict_to_object(self, buttons_dict):
        print("[Buttons Debug] Received dict:", buttons_dict)
        #normalize incoming keys to uppercase and handle multiple case formats
        bd = {}
        for k, v in buttons_dict.items():
            key = k.lower()
            bd[key] = v
    
        #set button states using lowercase attributes
        self.up = bd.get('up', False)
        self.down = bd.get('down', False)
        self.right = bd.get('right', False)
        self.left = bd.get('left', False)
        self.select = bd.get('select', False)
        self.start = bd.get('start', False)
        self.Y = bd.get('y', False)
        self.B = bd.get('b', False)
        self.X = bd.get('x', False)
        self.A = bd.get('a', False)
        self.L = bd.get('l', False)
        self.R = bd.get('r', False)

    def object_to_dict(self):
        return {
        'Up': self.up,
        'Down': self.down,
        'Right': self.right,
        'Left': self.left,
        'Select': self.select,
        'Start': self.start,
        'Y': self.Y,
        'B': self.B,
        'X': self.X,
        'A': self.A,
        'L': self.L,
        'R': self.R
    }