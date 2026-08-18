import os

# Anchored to this file's location (two levels up), so every artifact path below
# resolves the same no matter which directory the entry-point launcher is invoked
# from. Launchers are run from the repo root as `python project.py`, but anchoring to
# __file__ rather than CWD means a stray `cd` can't silently redirect where networks,
# replay buffers, and logs are read/written.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

NETWORK_PATH = os.path.join(DATA_DIR, 'Network.keras')
NETWORKS_DIR = os.path.join(DATA_DIR, 'Networks')
TRAINING_EXAMPLES_DIR = os.path.join(DATA_DIR, 'TrainingExamples')
CURRENT_TRAINING_EXAMPLES_PATH = os.path.join(TRAINING_EXAMPLES_DIR, 'current_training_examples.pkl')

LOGS_DIR = os.path.join(DATA_DIR, 'logs')
TRAINING_LOG_PATH = os.path.join(LOGS_DIR, 'training_log.csv')
GATING_LOG_PATH = os.path.join(LOGS_DIR, 'gating_log.csv')
CONFIG_HISTORY_DIR = os.path.join(LOGS_DIR, 'config_history')
CONSOLE_LOG_DIR = os.path.join(LOGS_DIR, 'console')
EPOCH_HISTORY_DIR = os.path.join(LOGS_DIR, 'epoch_history')

TOURNAMENT_LOG_PATH = os.path.join(LOGS_DIR, 'tournament_log.csv')
TOURNAMENT_SNAPSHOT_ROOT = os.path.join(NETWORKS_DIR, 'tournament_snapshots')


def ensure_data_dirs():
    for directory in (DATA_DIR, NETWORKS_DIR, TRAINING_EXAMPLES_DIR, LOGS_DIR,
                       CONFIG_HISTORY_DIR, CONSOLE_LOG_DIR, EPOCH_HISTORY_DIR):
        os.makedirs(directory, exist_ok=True)
