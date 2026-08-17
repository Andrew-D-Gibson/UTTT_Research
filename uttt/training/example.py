import numpy as np

class TrainingExample:
    def __init__(self, board, MCTS_pi):
        # Copy the board values, and flip who's to play if necessary
        self.board_array = board.get_array_representation()
  
        # Copy the search probabilities, re-assigning them to match their position in the board
        self.search_probs = np.zeros(81)
        
        for i, move in enumerate(board.find_moves()):
            self.search_probs[move] = MCTS_pi[i]
            
        
    def add_reward(self, reward):
        if self.board_array[0,0,2]: # If it's x's move
            self.reward = np.array([reward])
        else:
            self.reward = np.array([-reward])