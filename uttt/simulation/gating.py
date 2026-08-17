import tensorflow as tf

from uttt.player.agent import ProbabilisticNetworkMCTSAgent, agent_match
from uttt.worker import configure_cpu_worker


def simulate_gating_games(candidate_path, champion_path, num_games, start_index, progress_queue=None):
    configure_cpu_worker()

    # compile=False: gating never calls fit(), so building the optimizer/loss
    # graph on load is pure waste - and this worker loads two full models
    # (candidate + champion) instead of self-play's one.
    candidate_network = tf.keras.models.load_model(candidate_path, compile=False)
    champion_network = tf.keras.models.load_model(champion_path, compile=False)

    candidate_agent = ProbabilisticNetworkMCTSAgent(candidate_network)
    champion_agent = ProbabilisticNetworkMCTSAgent(champion_network)

    return agent_match(candidate_agent, champion_agent, num_of_games=num_games,
                        start_index=start_index, progress_queue=progress_queue)
