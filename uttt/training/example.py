import numpy as np

class TrainingExample:
    def __init__(self, board, MCTS_pi):
        # Copy the board values, and flip who's to play if necessary
        self.board_array = board.get_array_representation()

        # board_array is canonicalized to mover/opponent, so it no longer carries an
        # explicit whose-move-was-it channel - record it separately for add_reward().
        self.x_to_move = board.x_move

        # Copy the search probabilities, re-assigning them to match their position in the board
        self.search_probs = np.zeros(81)

        for i, move in enumerate(board.find_moves()):
            self.search_probs[move] = MCTS_pi[i]


    def add_reward(self, reward):
        if self.x_to_move:
            self.reward = np.array([reward])
        else:
            self.reward = np.array([-reward])