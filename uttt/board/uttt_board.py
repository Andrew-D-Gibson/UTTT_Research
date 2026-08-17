import numpy as np


# 10000 random games:
# 24.75 sec pre-optimization
# 10.87 sec post-optimization
# HELL YEAH

class UTTTBoard:
    def __init__(self):
        # Clear the boards
        self.x = 0 # 81 bits
        self.o = 0
        self.eligible_subboards = [0,1,2,3,4,5,6,7,8]

        self.subboard_x = 0b000000000
        self.subboard_o = 0b000000000
        self.subboard_draws = 0b000000000

        # Set x to move first
        self.x_move = True;

        # Record the value of the board if it's a win for x (1), loss for x (-1), or a draw (0)
        self.value = 0

        # Undo stack for make_move()/unmake_move(), so a single board can be walked
        # forward and backward during tree search instead of being copied per node.
        self._history = []


    def find_moves(self):
        moves = []
        for subboard in self.eligible_subboards:
            for i in range(9):
                if self.check_empty((subboard*9) + i):
                    moves.append((subboard*9) + i)         
        return moves
    
    
    def make_move(self, move: int):
        if int(move/9) not in self.eligible_subboards:
            print("ERROR: Trying to move in ineligible subboard.")
            return

        if not self.check_empty(move):
            print("ERROR: Trying to play in occupied square.")
            return

        # Snapshot just enough state to exactly undo this move. eligible_subboards
        # is always *replaced* wholesale below (never mutated in place), so it's
        # safe to keep a reference to the old list rather than copying it.
        self._history.append((
            move,
            self.x_move,
            self.eligible_subboards,
            self.subboard_x,
            self.subboard_o,
            self.subboard_draws,
        ))

        if (self.x_move):
            self.x |= 2**move
        else:
            self.o |= 2**move

        self.x_move = not self.x_move

        # Check if the subboard we just played on is undecided (still playable)
        self.update_subboard_game_over(int(move/9))

        # Check if we're sending to an already decided subboard
        already_decided_subboards = self.subboard_x | self.subboard_o | self.subboard_draws
        if 2**(move%9) & already_decided_subboards:
            self.eligible_subboards = []
            for i in range(9):
                if not (2**i) & already_decided_subboards:
                    self.eligible_subboards.append(i)
        else:
            self.eligible_subboards = [move%9]


    def unmake_move(self):
        move, x_move, eligible_subboards, subboard_x, subboard_o, subboard_draws = self._history.pop()

        # x_move records whose turn it was *before* the move, i.e. who made it.
        if x_move:
            self.x &= ~(2**move)
        else:
            self.o &= ~(2**move)

        self.x_move = x_move
        self.eligible_subboards = eligible_subboards
        self.subboard_x = subboard_x
        self.subboard_o = subboard_o
        self.subboard_draws = subboard_draws


    def check_empty(self, move: int):
        return not (2**move) & (self.x | self.o)
    

    def update_subboard_game_over(self, subboard: int):
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
            combo = combo << subboard * 9

            if self.x & combo == combo:
                # X won
                self.subboard_x |= 2**subboard
                return True
            elif self.o & combo == combo:
                # O won
                self.subboard_o |= 2**subboard
                return True
        
        # No player has won, so check for valid moves (squares still open)
        full_subboard = 0b111111111 << (subboard * 9)
        if full_subboard & (self.x | self.o) == full_subboard:
            # There are no open squares, so it's a draw
            self.subboard_draws |= 2**subboard
            return True
        
        # No player has won and it's not a draw, so the game is not over
        return False
    
    
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
            if self.subboard_x & combo == combo:
                # X won
                self.value = 1
                return True
            elif self.subboard_o & combo == combo:
                # O won
                self.value = -1
                return True
        
        # No player has won, so check for valid places to move
        if not self.eligible_subboards:
            # There are no possible moves, so it's a draw
            # self.value defaults to 0, so no need to change it
            return True
        
        # No player has won and it's not a draw, so the game is not over
        return False
    
    
    def get_array_representation(self):
        array = np.empty((9,9,4))

        # Player positions
        array[:,:,0] = self.symbol_array_representation(self.x)
        array[:,:,1] = self.symbol_array_representation(self.o)

        # Whose move it is
        if self.x_move:
            array[:,:,2] = np.ones((9,9))
        else:
            array[:,:,2] = np.zeros((9,9))
            
        # Which subboards are eligible for moving in
        eligible_subboards_bitboard = 0 # 81 bits
        for subboard in self.eligible_subboards:
            eligible_subboards_bitboard |= 0b111111111 << (subboard * 9)
        array[:,:,3] = self.symbol_array_representation(eligible_subboards_bitboard)
        
        return array
    
    
    def symbol_array_representation(self, symbol_bitboard):
        # Vectorized bit-unpack. symbol_bitboard's bit k is the value at move
        # k = subboard*9 + local_cell (subboard, local_cell each 0-8, row-major:
        # row = idx // 3, col = idx % 3 - see the win_combos bit layout above).
        # np.unpackbits(..., bitorder='little') gives bits[k] == (symbol_bitboard >> k) & 1
        # for the whole 81-bit value in one vectorized call, avoiding the
        # bin()-string-parsing + per-subboard np.pad/np.resize/np.append loop
        # this replaces (that loop runs on every MCTS leaf evaluation).
        bits = np.unpackbits(
            np.frombuffer(symbol_bitboard.to_bytes(11, 'little'), dtype=np.uint8),
            bitorder='little',
        )[:81]

        # Reshape to (subboard_row, subboard_col, local_row, local_col), interleave
        # subboard/local rows and cols into a 9x9 grid, then flip both axes to match
        # this method's original (and still relied-upon) row/col orientation.
        board = bits.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
        return board[::-1, ::-1]
    
    
    def print(self):
        array = self.get_array_representation()
            
        for i in range(9):
            for j in range(9):
                if array[i,j,0]:
                    print(' x ', end='')
                elif array[i,j,1]:
                    print(' o ', end='')
                else:
                    print(' . ', end ='')
                
                if (j+1)%3 == 0 and j != 8:
                    print( ' | ', end='')
                    
            if i == 3 or i == 4 or i == 5:
                if i == 4:
                    print('    =>    ', end ='')
                else:
                    print('          ', end ='')
                    
                for k in range(2,-1,-1):
                    if self.subboard_x & (2**(3*(5-i) + k)):
                        print(' x ', end='')
                    elif self.subboard_o & (2**(3*(5-i) + k)):
                        print(' o ', end='')
                    else:
                        print(' . ', end='')
                        
            if i == 6:
                print('            ', end='')
                if self.is_game_over():
                    if (self.value == 1):
                        print('X Win!', end='')
                    elif (self.value == -1):
                        print('O Win!', end='')
                    else: 
                        print('Draw!', end='')      
                    
            print(' ')
            
            if (i+1)%3 == 0 and i != 8:
                print('-'*33)
                
        if self.x_move:
            print('X\'s Move     ', end='')
        else:
            print('O\'s Move     ', end='')
        
        print('Eligible Sub-boards: ', end='')
        print(self.eligible_subboards)
        print("---")