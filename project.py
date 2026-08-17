# Entry point for the self-play training loop. Runs from the repo root.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from uttt.paths import NETWORK_PATH, ensure_data_dirs
from uttt.network.architectures import NetworkArchitectureTester
from uttt.training.manager import TrainingManager

if __name__ == '__main__':
    ensure_data_dirs()
    if not os.path.exists(NETWORK_PATH):
        model = NetworkArchitectureTester.convNet()
        model.save(NETWORK_PATH)

    trainer = TrainingManager()
    trainer.train()
