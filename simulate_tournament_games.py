import os
import time

import tensorflow as tf

from PlayerAgent import ProbabilisticNetworkMCTSAgent, RolloutMCTSAgent, agent_game
from worker_utils import configure_cpu_worker


# Rebuilt once per worker process by init_tournament_worker (a multiprocessing.Pool
# initializer), not once per game - mirrors simulate_gating_games.py loading its two
# models once per worker rather than once per game. Every worker independently builds
# its own copy of every agent, in the same order as the main process's rating list, so
# agent_1_idx/agent_2_idx line up across processes without any state being shared.
_agents = None


def init_tournament_worker(agent_specs):
    global _agents
    configure_cpu_worker()

    _agents = []
    for spec in agent_specs:
        if spec['kind'] == 'network':
            # compile=False: tournament games never call fit(), so building the
            # optimizer/loss graph on load is pure waste (same reasoning as
            # simulate_gating_games.py).
            network = tf.keras.models.load_model(spec['path'], compile=False)
            agent = ProbabilisticNetworkMCTSAgent(network)
        else:
            agent = RolloutMCTSAgent(iterations=spec['depth'])
        agent.name = spec['name']
        _agents.append(agent)


def play_tournament_game(agent_1_idx, agent_2_idx, game_index, progress_queue=None):
    # agent_game() already calls reset() on both agents internally, so this worker's
    # agents are safe to reuse for its next assigned game with no leftover MCTS state
    # (same invariant simulate_gating_games.py/agent_match rely on).
    start_time = time.time()
    result = agent_game(_agents[agent_1_idx], _agents[agent_2_idx])
    duration = time.time() - start_time

    payload = (os.getpid(), agent_1_idx, agent_2_idx, game_index, result, duration)
    if progress_queue is not None:
        progress_queue.put(payload)
    return payload
