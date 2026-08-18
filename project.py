# Entry point for the self-play training loop. Runs from the repo root.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from uttt.paths import NETWORK_PATH, ensure_data_dirs
from uttt.network.architectures import convNet
from uttt.training.manager import TrainingManager
from uttt.run_logging import new_session_id, start_console_log

if __name__ == '__main__':
    ensure_data_dirs()
    session_id = new_session_id()
    start_console_log(session_id)

    if not os.path.exists(NETWORK_PATH):
        model = convNet()
        model.save(NETWORK_PATH)

    trainer = TrainingManager(session_id=session_id)
    trainer.train()
