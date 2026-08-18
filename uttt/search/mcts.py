import numpy as np
import random

from uttt.board.uttt_board import UTTTBoard

from uttt.config import config


class MCTS():
    # Only the root of a tree owns a real, persistent UTTTBoard (the actual game
    # state). Every other node is just search statistics (t, n, children,
    # search_probs) keyed by the move that reaches it. During search, a single
    # shared board is walked forward with board.make_move() as we descend and
    # walked back with board.unmake_move() as we return, instead of every node
    # deep-copying the board it represents.
    def __init__(self, network = None, board = None, parent = None, move = None):
        self.reset()

        if network == None:
            self.rollout_func = self.random_rollout
        else:
            self.rollout_func = self.check_network

        self.network = network
        self.parent = parent
        self.move = move

        # Only ever passed for the root (a fresh game) or when make_move() hands
        # the now-advanced shared board over to whichever node becomes the new root.
        if board is not None:
            self.board = board


    def reset(self):
         # Set the node's total value to 0
        self.t = 0

        # Set the node's total visits to 0
        self.n = 0

        # Children nodes (also of class NNetMCTS)
        self.children = []

        # Search probabilities
        self.pi = []
        self.search_probs = []

        # Current board position (only meaningful for the root - see __init__)
        self.board = UTTTBoard()

        self.parent = None
        self.move = None


    def random_rollout(self, board):
        # Check if we're in a terminal position (game is won, lost, or a draw)
        # If it is, backpropagate the value up the tree
        # We 'absolute value' the return to properly punish this node for losing
        # (e.g. A "Game Over" when it's your turn is either a draw or loss, so -1 or 0 reward)
        if board.is_game_over():
            self.backpropagate(-np.abs(board.value))
            return

        legal_moves = board.find_moves()
        self.search_probs = np.ones(len(legal_moves)) / len(legal_moves)    # We treat all children equally in random rollout

        # Only materialize the one child we're actually going to visit instead of
        # deep-copying the board for every legal move just to discard all but one.
        chosen_move = random.choice(legal_moves)
        child = MCTS(network=self.network, parent=self, move=chosen_move)

        board.make_move(chosen_move)
        child.random_rollout(board)
        board.unmake_move()


    def check_network(self, board):
        # Check if we're in a terminal position (game is won, lost, or a draw)
        # If it is, backpropagate the value up the tree
        # We 'absolute value' the return to properly punish this node for losing
        # (e.g. A "Game Over" when it's your turn is either a draw or loss, so -1 or 0 reward)
        if board.is_game_over():
            self.backpropagate(-np.abs(board.value))
            return

        # Now we know the game isn't over, so return the neural network's evaluation.
        # Call the model directly instead of .predict() - .predict() rebuilds a
        # tf.data pipeline (plus other bookkeeping) on every call, which dominates
        # runtime when called once per MCTS iteration. Direct __call__ skips that
        # overhead while staying in normal eager mode.
        board_arrays = np.array([board.get_array_representation()])

        outputs = self.network(board_arrays, training=False)
        # np.asarray rather than .numpy(): outputs may be a real EagerTensor (in-process
        # network) or a plain numpy array (InferenceClient, uttt/inference/server.py) -
        # this works with either without MCTS needing to know which it's talking to.
        search_probs = np.asarray(outputs['policy_output'])
        value_est = np.asarray(outputs['value_output'])

        legal_logits = np.squeeze(search_probs)[board.find_moves()]  # Get only the logits for valid moves
        # Softmax over legal moves only (masking then softmaxing full-board logits
        # would let illegal moves' logits skew the normalization); subtract the max
        # first for numerical stability, standard log-sum-exp trick.
        legal_logits -= legal_logits.max()
        exp_logits = np.exp(legal_logits)
        self.search_probs = exp_logits / exp_logits.sum()

        value_est = np.squeeze(value_est)

        self.backpropagate(value_est)


    # Top level search method
    def search(self, iterations = None, add_root_noise = False):
        if iterations is None:
            iterations = config['mcts']['search_depth']

        # The root's own priors (self.search_probs) only exist after its first
        # visit, so spend one iteration evaluating it before mixing in noise -
        # this is exactly the iteration that would've happened first anyway.
        if self.n == 0:
            self.MCTS_iteration(self.board)
            iterations -= 1

        if add_root_noise and len(self.search_probs) > 0:
            epsilon = config['mcts']['dirichlet_epsilon']
            noise = np.random.dirichlet([config['mcts']['dirichlet_alpha']] * len(self.search_probs))
            self.search_probs = (1 - epsilon) * self.search_probs + epsilon * noise

        for i in range(iterations):
            self.MCTS_iteration(self.board)

        # Find the number of visits (n) for each node as a proxy for goodness of the node
        self.pi = [child.n / (self.n - 1) for child in self.children]


    def make_move(self, move=None):
        if move not in self.board.find_moves():
            print('No valid move given!')
            return

        # Permanently commit this move on the shared board - this becomes the
        # actual next game state, not a hypothetical explored during search.
        self.board.make_move(move)

        for child in self.children:
            if child.move == move:
                child.board = self.board
                child.parent = None
                return child

        new_node = MCTS(network=self.network, board=self.board, parent=None, move=move)
        return new_node


    # Returns a list of all the valid children nodes for the given (already-current) board
    def make_children_list(self, board):
        return [MCTS(network=self.network, parent=self, move=move) for move in board.find_moves()]


    # Backpropagates a states value back up through all the parent nodes,
    # flipping it's sign every time b/c of the tree's adversarial nature.
    # (e.g. a good node for O is a bad node for X)
    def backpropagate(self, value):
        self.t += value
        self.n += 1

        if self.parent != None:
            self.parent.backpropagate(-value)


    # Performs a single iteration of a Monte-Carlo tree search.
    # `board` is always the shared board, already advanced to reflect `self`'s
    # position - any recursive descent into a child must make/unmake around it.
    def MCTS_iteration(self, board):
        # Check if we're at a terminal node (no more game to play because it's over!)
        # If so, then just backpropagate the value of the position
        if board.is_game_over():
            self.backpropagate(-np.abs(board.value))    # Prior node won or drew, so reward it with +1 (or 0 so w/e)
            return

        # Check if we're at a leaf node (no children exist in this tree)
        if self.children == []:
            # If we've never been here before (n = 0) then perform rollout
            if self.n == 0:
                self.rollout_func(board)

            # Otherwise extend the tree by finding all the children and then
            # performing rollout on the first child
            else:
                self.children = self.make_children_list(board)

                board.make_move(self.children[0].move)
                self.children[0].rollout_func(board)
                board.unmake_move()

        # The game isn't over and we're not at a leaf node, therefore we're somewhere in the search tree.
        # Pick the child that maximises UCB and continue search
        else:
            children_UCB = []

            for child, search_prob in zip(self.children, self.search_probs):
                value_term = child.t / (child.n or 1) # Average value (total value / number of times visited)
                exploration_term = search_prob * (np.sqrt(self.n) / (1 + child.n))
                children_UCB.append(-value_term + (config['mcts']['exploration_parameter'] * exploration_term))
                # The -value_term is because this node is its child node's opponent

            best_UCB = np.argmax(children_UCB)
            best_child = self.children[best_UCB]

            board.make_move(best_child.move)
            best_child.MCTS_iteration(board)
            board.unmake_move()
