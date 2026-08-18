# Entry point for the self-play training loop. Runs from the repo root.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from uttt.paths import NETWORK_PATH, ensure_data_dirs
from uttt.network.architectures import build_network
from uttt.training.manager import TrainingManager
from uttt.run_logging import new_session_id, start_console_log
from uttt.worker import enable_gpu_memory_growth

if __name__ == '__main__':
    ensure_data_dirs()
    session_id = new_session_id()
    start_console_log(session_id)

    # Must happen before build_network() below (or anything else that touches a GPU) -
    # TF refuses to change memory growth settings after a device's first use, and a
    # fresh run's build_network() call is exactly that first use in this process.
    enable_gpu_memory_growth()

    if not os.path.exists(NETWORK_PATH):
        model = build_network()
        model.save(NETWORK_PATH)

    trainer = TrainingManager(session_id=session_id)
    trainer.train()
