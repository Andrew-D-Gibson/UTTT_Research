import os
import time
import numpy as np

from uttt.search.mcts import MCTS
from uttt.training.example import TrainingExample
from uttt.worker import seed_worker_rng
from uttt.inference.server import InferenceClient

from uttt.config import config

# Set once per worker process by init_self_play_worker (a multiprocessing.Pool
# initializer), not passed as a per-call argument - a raw multiprocessing.Queue can
# only be handed to a worker at process-creation time (Pool's initargs), not pickled
# through Pool's per-task dispatch queue (apply_async args), which raises
# "Queue objects should only be shared between processes through inheritance".
# Mirrors uttt/simulation/tournament.py's _agents/init_tournament_worker pattern.
# One queue per distinct GPU (see TrainingManager.start_inference_servers) - InferenceClient
# round-robins across them so load actually splits evenly across GPUs.
_request_queues = None


def init_self_play_worker(request_queues):
    global _request_queues
    _request_queues = request_queues
    seed_worker_rng()


def simulate_self_play_games(progress_queue=None):
    training_examples = []

    network = InferenceClient(_request_queues)
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
