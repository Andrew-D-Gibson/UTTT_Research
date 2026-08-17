# Entry point for the baseline Elo calibration: round-robins a fixed roster of random /
# rollout-MCTS agents at several search depths. Runs from the repo root.
from uttt.evaluation.elo import ELOManager
from uttt.player.agent import RandomAgent, RolloutMCTSAgent

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    iterations = 20

    elo = ELOManager([
        RandomAgent(),
        RolloutMCTSAgent(iterations=16),
        RolloutMCTSAgent(iterations=64),
        RolloutMCTSAgent(iterations=256),
        RolloutMCTSAgent(iterations=1024),
        RolloutMCTSAgent(iterations=4096),
    ])

    history = elo.calibrate(iterations)

    print(history)
