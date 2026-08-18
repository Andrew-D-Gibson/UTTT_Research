from uttt.player.agent import ProbabilisticNetworkMCTSAgent, agent_match
from uttt.worker import seed_worker_rng
from uttt.inference.server import InferenceClient

# Set once per worker process by init_gating_worker (a multiprocessing.Pool
# initializer) - see uttt/simulation/self_play.py's identical _request_queue for why
# these can't be passed as simulate_gating_games() arguments instead.
_candidate_queue = None
_champion_queue = None


def init_gating_worker(candidate_queue, champion_queue):
    global _candidate_queue, _champion_queue
    _candidate_queue = candidate_queue
    _champion_queue = champion_queue
    seed_worker_rng()


def simulate_gating_games(num_games, start_index, progress_queue=None):
    # Candidate and champion are different weights, served by two separate
    # inference-server processes (see TrainingManager.run_gating) - one InferenceClient
    # per queue routes each side's requests to the right one.
    candidate_network = InferenceClient(_candidate_queue)
    champion_network = InferenceClient(_champion_queue)

    candidate_agent = ProbabilisticNetworkMCTSAgent(candidate_network)
    champion_agent = ProbabilisticNetworkMCTSAgent(champion_network)

    return agent_match(candidate_agent, champion_agent, num_of_games=num_games,
                        start_index=start_index, progress_queue=progress_queue)
