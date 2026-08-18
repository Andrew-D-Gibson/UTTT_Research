import os
import sys
import json
from datetime import datetime

from uttt.config import config
from uttt.paths import CONFIG_HISTORY_DIR, CONSOLE_LOG_DIR


def snapshot_config():
    # Writes the live config to config_history/config_v{version}.json the first time this
    # version number is seen, so any episode's exact hyperparameters can be recovered later
    # by its logged config_version, even after config.py has since moved on. If a snapshot
    # for this version already exists but no longer matches, that means config.py was edited
    # without bumping 'version' - warn rather than silently letting the audit trail drift.
    version = config['version']
    snapshot_path = os.path.join(CONFIG_HISTORY_DIR, f'config_v{version}.json')

    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r') as file:
            saved = json.load(file)
        if saved != config:
            print(f'WARNING: {snapshot_path} does not match the current config - '
                  f"did you forget to bump config['version'] in uttt/config.py?")
        return

    with open(snapshot_path, 'w') as file:
        json.dump(config, file, indent=2)
    print(f'Snapshotted config version {version} to {snapshot_path}')


class _Tee:
    # Duplicates every write to both the original stream and a log file, so print()
    # calls throughout the codebase stay visible in the terminal while also being
    # captured durably for a long unattended run.
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def new_session_id():
    # One identifier per process lifetime, shared between the console log filename and
    # the session_id column stamped on every training_log.csv/gating_log.csv row - an
    # exact join between a logged episode and the raw console output from that run,
    # instead of having to eyeball timestamps against each other.
    return datetime.now().strftime('%Y-%m-%d_%H%M%S')


def start_console_log(session_id):
    # Redirects sys.stdout/sys.stderr through a tee so everything printed this process
    # lifetime - self-play/gating progress, training metrics, promotion verdicts, Keras's
    # own fit() output - also lands in one file per run under data/logs/console/, named by
    # session_id so a restart never collides with or overwrites a prior run's log.
    log_path = os.path.join(CONSOLE_LOG_DIR, f'run_{session_id}.log')
    log_file = open(log_path, 'a')

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print(f'Logging console output to {log_path}')
    return log_path
