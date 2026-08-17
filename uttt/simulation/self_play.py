import os
import time
import numpy as np
import tensorflow as tf

from uttt.search.mcts import MCTS
from uttt.training.example import TrainingExample
from uttt.worker import configure_cpu_worker
from uttt.paths import NETWORK_PATH

from uttt.config import config


def simulate_self_play_games(progress_queue=None):
    training_examples = []

    configure_cpu_worker()

    network = tf.keras.models.load_model(NETWORK_PATH)
    mcts = MCTS(network = network)

    num_games = config['self_play']['num_of_self_play_games_per_process']
    for i in range(num_games):
        game_start_time = time.time()
        new_training_examples = []
        ply = 0

        while not mcts.board.is_game_over():
            mcts.search(add_root_noise=True)

            new_training_examples.append(TrainingExample(mcts.board, mcts.pi))

            # Sample proportional to visit counts early in the game (encourages
            # opening diversity), then play greedily once the game has settled,
            # since noisy endgame play produces low-quality training signal.
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
            # Report through the shared queue so the main process can print one
            # consolidated, ETA'd progress stream instead of num_of_processes workers
            # each writing to stdout independently and interleaving mid-line.
            progress_queue.put((os.getpid(), i + 1, num_games, ply, duration))
        else:
            print(f'[worker {os.getpid()}] finished game {i+1}/{num_games} ({ply} plies, {duration:.1f}s)', flush=True)

    return training_examples
