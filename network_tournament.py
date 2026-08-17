# Entry point for the network tournament: snapshots a ladder of checkpoint networks plus
# raw-MCTS baselines and runs a parallelized Elo round-robin. Runs from the repo root.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from uttt.evaluation.network_tournament import run_default

if __name__ == '__main__':
    run_default()
