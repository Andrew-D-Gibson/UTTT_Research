import random
import numpy as np
import os
import time
import tensorflow as tf

from uttt.paths import NETWORK_PATH
from uttt.board.uttt_board import UTTTBoard
from uttt.search.mcts import MCTS

from uttt.config import config

class PlayerAgent:
    def __init__(self):
        self.elo = 1200
        self.name = ''

    def get_move(self):
        pass

    def make_move(self, move):
        pass
    
    def reset(self):
        pass
    

class RandomAgent(PlayerAgent):
    def __init__(self):
        super().__init__()
        self.name = 'Random Agent'
        self.board = UTTTBoard()
    
    def get_move(self):
        return random.choice(self.board.find_moves())
    
    def make_move(self, move):
        self.board.make_move(move)
        
    def reset(self):
        self.board = UTTTBoard()
        

class RolloutMCTSAgent(PlayerAgent):
    def __init__(self, iterations = None):
        super().__init__()

        if iterations is None:
            self.iterations = config['mcts']['search_depth']
        else:
            self.iterations = iterations

        self.name = f'MCTS with depth {self.iterations}'
        self.mcts = MCTS()
    
    def get_move(self):
        self.mcts.search(iterations=self.iterations)
        child_choice = np.argmax(self.mcts.pi)        
        return self.mcts.children[child_choice].move
    
    def make_move(self, move):
        self.mcts = self.mcts.make_move(move)
        
    def reset(self):
        self.mcts.reset()
        
        
class NetworkMCTSAgent(PlayerAgent):
    def __init__(self, network):
        super().__init__()
        self.name = 'AlphaZero'
        self.mcts = MCTS(network=network)
    
    def get_move(self):
        self.mcts.search()
        child_choice = np.argmax(self.mcts.pi)
        return self.mcts.children[child_choice].move
    
    def make_move(self, move):
        self.mcts = self.mcts.make_move(move)
        
    def reset(self):
        self.mcts.reset()
        

class ProbabilisticNetworkMCTSAgent(PlayerAgent):
    def __init__(self, network):
        super().__init__()
        self.name = 'Probabilistic AlphaZero'
        self.mcts = MCTS(network=network)
    
    def get_move(self):
        self.mcts.search()
        child_choice = np.random.choice(len(self.mcts.pi), p=self.mcts.pi)
        return self.mcts.children[child_choice].move
    
    def make_move(self, move):
        self.mcts = self.mcts.make_move(move)
        
    def reset(self):
        self.mcts.reset()
        
        
class RawNetworkAgent(PlayerAgent):
    def __init__(self, network):
        super().__init__()
        self.name = 'Raw AlphaZero Network'
        self.network = network
        self.board = UTTTBoard()
    
    def get_move(self):
        board_arrays = np.array([self.board.get_array_representation()])
        search_probs, value_est = self.network(board_arrays, training=False)
        search_probs = np.squeeze(search_probs.numpy())[self.board.find_moves()]

        return self.board.find_moves()[np.argmax(search_probs)]
    
    def make_move(self, move):
        self.board.make_move(move)
        
    def reset(self):
        self.board = UTTTBoard()


def agent_game(x_agent, o_agent):
    board = UTTTBoard()
    
    while True:
        x_move = x_agent.get_move()
        
        x_agent.make_move(x_move)
        o_agent.make_move(x_move)
        board.make_move(x_move)
        
        if board.is_game_over():
            break
        
        
        o_move = o_agent.get_move()
        
        x_agent.make_move(o_move)
        o_agent.make_move(o_move)
        board.make_move(o_move)
        
        if board.is_game_over():
            break
            
    x_agent.reset()
    o_agent.reset()
    return board.value
        
        
def agent_match(agent_1, agent_2, num_of_games, start_index=0, progress_queue=None):
    agent_1_wins = 0
    draws = 0
    agent_2_wins = 0

    start_time = time.time()
    for i in range(num_of_games):
        game_start = time.time()

        # start_index lets a chunk of games run inside a worker process while
        # still alternating colors exactly as if it were a slice of one long
        # sequential match (see uttt/simulation/gating.py) - restarting i%2 at 0
        # in every chunk would systematically favor whichever agent goes first.
        if (start_index + i) % 2 == 0:
            game_result = agent_game(agent_1, agent_2)
        else:
            game_result = -agent_game(agent_2, agent_1)

        if game_result == 1:
            agent_1_wins += 1
            outcome = 'Agent 1 win'
        elif game_result == -1:
            agent_2_wins += 1
            outcome = 'Agent 2 win'
        else:
            draws += 1
            outcome = 'Draw'

        duration = time.time() - game_start

        if progress_queue is not None:
            # Mirrors simulate_self_play_games' queue-based progress reporting -
            # avoids multiple gating worker processes interleaving prints, and lets
            # the caller compute aggregate elapsed/ETA across all chunks instead of
            # this chunk-local (and therefore misleading) one.
            progress_queue.put((os.getpid(), i + 1, num_of_games, outcome, duration))
        else:
            elapsed = time.time() - start_time
            eta = (elapsed / (i+1)) * (num_of_games - (i+1))
            print(f'  game {i+1}/{num_of_games}: {outcome} ({duration:.1f}s) '
                  f'- tally {agent_1_wins}/{draws}/{agent_2_wins}, ETA ~{eta:.0f}s')

    return agent_1_wins, draws, agent_2_wins
    

def test_network_vs_mcts():
    print('Baseline: network+MCTS (Agent 1) vs rollout MCTS (Agent 2):')
    network = tf.keras.models.load_model(NETWORK_PATH)
    wins, draws, losses = agent_match(NetworkMCTSAgent(network),
                                          RolloutMCTSAgent(),
                                          num_of_games=config['self_play']['num_of_baseline_games'])
    print(f'W/D/L: {wins} / {draws} / {losses}')


def test_raw_network_vs_random():
    print('Baseline: raw network (Agent 1) vs random (Agent 2):')
    network = tf.keras.models.load_model(NETWORK_PATH)
    wins, draws, losses = agent_match(RawNetworkAgent(network),
                                          RandomAgent(),
                                          num_of_games=config['self_play']['num_of_baseline_games'])
    print(f'W/D/L: {wins} / {draws} / {losses}')