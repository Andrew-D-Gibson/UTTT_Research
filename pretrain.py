# Optional fire-and-forget bootstrap entry point: builds/resumes the raw-MCTS pool,
# then trains against it with no self-play or gating. Runs from the repo root.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from uttt.paths import NETWORK_PATH, ensure_data_dirs
from uttt.network.architectures import NetworkArchitectureTester
from uttt.training.generate_raw_pool import generate_pool
from uttt.training.pretrain_manager import PretrainManager

if __name__ == '__main__':
    ensure_data_dirs()
    if not os.path.exists(NETWORK_PATH):
        model = NetworkArchitectureTester.convNet()
        model.save(NETWORK_PATH)

    generate_pool()

    pretrainer = PretrainManager()
    pretrainer.pretrain()
