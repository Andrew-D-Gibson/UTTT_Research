import numpy as np

class TTTBoard:
    def __init__(self):
        # Clear the boards
        self.x = 0;
        self.o = 0;
        
        # Set x to move first
        self.x_move = True;
            
        # Record the value of the board if it's a win for x (1), loss for x (-1), or a draw/undecided (0)
        self.value = 0
        
        
    def make_move(self, square):
        if not self.check_empty(square):
            print("ERROR: Trying to play in a square that's already taken.")
            return
        
        if (self.x_move):
            self.x |= (2**square)
        else:
            self.o |= (2**square)
        
        self.x_move = not self.x_move
        
        
    def check_empty(self, square):
        return (2**square) & (self.x | self.o) == 0
    
    
    def find_moves(self):
        moves = []
        for i in range(9):
            if self.check_empty(i):
                moves.append(i)
                
        return moves
    
    
    def is_game_over(self):
        win_combos = [
            0b111000000, # Bottom row win
            0b000111000, # Middle row win
            0b000000111, # Top row win
            0b100100100, # Right column win
            0b010010010, # Middle column win
            0b001001001, # Left Column Win
            0b100010001, # Top left to bottom right diagonal win
            0b001010100  # Top right to bottom left diagonal win
        ]
        for combo in win_combos:
            if self.x & combo == combo:
                # X won
                self.value = 1
                return True
            elif self.o & combo == combo:
                # O won
                self.value = -1
                return True
        
        # No player has won, so check for any valid moves 
        # e.g. are there any squares still open?
        if 0b111111111 ^ (self.x | self.o) == 0:
            # There are no open squares, so it's a draw
            # self.value defaults to 0, so no need to change it
            return True
        
        # No player has won and it's not a draw, so the game is not over
        return False
    
    
    def print(self):
        for i in range(2,-1,-1):
            for j in range(2,-1,-1):
                if self.x & 2**((i*3) + j):
                    print(' x ', end = '')
                elif self.o & 2**((i*3) + j):
                    print(' o ', end = '')
                else:
                    print(' _ ', end = '')
                
            if (i == 1):
                if self.is_game_over():
                    if self.value == 1:
                        print('    X Won!', end = '')
                    elif self.value == -1:
                        print('    O Won!', end = '')
                    else:
                        print('    Draw!', end = '')
                elif (self.x_move):
                    print('    X\'s move', end = '')
                else:
                    print('    O\'s move', end = '')
                    
            print('\n')
        print('\n')
            
            
    def get_array_representation(self):
        array_representation = np.empty((3,3,3))
        
        x_array = [np.uint8(bit) for bit in bin(self.x)[2:]]
        x_array = np.pad(x_array, (9-len(x_array),0))
        array_representation[:,:,0] = np.resize(x_array, (3,3))
        
        o_array = [np.uint8(bit) for bit in bin(self.o)[2:]]
        o_array = np.pad(o_array, (9-len(o_array),0))
        array_representation[:,:,1] = np.resize(o_array, (3,3))
        
        if self.x_move:
            array_representation[:,:,2] = np.ones((3,3))
        else:
            array_representation[:,:,2] = np.zeros((3,3))
        
        return array_representation