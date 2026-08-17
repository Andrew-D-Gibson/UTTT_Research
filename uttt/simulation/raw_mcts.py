import os
import random
import time
import numpy as np

from uttt.search.mcts import MCTS
from uttt.training.example import TrainingExample

from uttt.config import config


def simulate_raw_mcts_games(progress_queue=None):
    # Network-free counterpart to uttt/simulation/self_play.py: MCTS(network=None)
    # falls back to random-rollout evaluation, so this generates the same
    # TrainingExample shape (board_array, search_probs, reward) without ever
    # touching Network.keras or TensorFlow. Deliberately skips
    # worker_utils.configure_cpu_worker() - that helper's GPU/thread config calls
    # would themselves force a TF import in a worker that otherwise never needs one.
    random.seed(os.getpid())
    np.random.seed(os.getpid())

    training_examples = []

    mcts = MCTS(network=None)
    depth = config['mcts']['pretrain_mcts_depth']

    num_games = config['self_play']['num_of_self_play_games_per_process']
    for i in range(num_games):
        game_start_time = time.time()
        new_training_examples = []
        ply = 0

        while not mcts.board.is_game_over():
            mcts.search(iterations=depth)

            new_training_examples.append(TrainingExample(mcts.board, mcts.pi))

            # Same opening-diversity schedule as self-play: sample proportional to
            # visit counts early, then play greedily.
            if ply < config['self_play']['temperature_moves']:
                child_choice = np.random.choice(len(mcts.pi), p=mcts.pi)
            else:
                child_choice = np.argmax(mcts.pi)

            mcts = mcts.make_move(mcts.children[child_choice].move)
            ply += 1

        for example in new_training_examples:
            example.add_reward(mcts.board.value)

        training_examples.extend(new_training_examples)
        mcts.reset()

        duration = time.time() - game_start_time
        if progress_queue is not None:
            progress_queue.put((os.getpid(), i + 1, num_games, ply, duration))
        else:
            print(f'[worker {os.getpid()}] finished game {i+1}/{num_games} ({ply} plies, {duration:.1f}s)', flush=True)

    return training_examples
