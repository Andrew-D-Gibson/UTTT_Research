config = {
    # Bump this by hand any time you change a value below. It gets snapshotted once to
    # data/logs/config_history/config_v{version}.json and stamped on every training_log.csv/
    # gating_log.csv row, so a given episode's exact hyperparameters can always be
    # recovered later, even if config.py has since moved on (see uttt/run_logging.py).
    'version': 1,

    'network': {
        'architecture': 'hierarchicalResNet',   # 'hierarchicalResNet' or 'convNet' - see uttt/network/architectures.py's build_network()
    },

    'mcts': {
        'exploration_parameter': 1.5,
        'search_depth': 256,           
        'dirichlet_alpha': 1.2,       # root exploration noise (AlphaZero-style)
        'dirichlet_epsilon': 0.25,    # weight of noise vs. network prior at the root
    },

    'inference': {
        # One inference server process per entry - repeat a GPU id to run multiple server
        # processes sharing that GPU, e.g. [0, 0, 0, 0, 1, 1, 1, 1] for 4 servers per GPU on
        # a 2-GPU box. This matters because each server's request/response dispatch loop
        # (multiprocessing.Queue.get + Pipe.send, one round trip per leaf evaluation) is
        # single-threaded Python and has real per-message overhead - at high worker counts
        # that loop saturates well before the GPU does (symptom: GPU usage stays low while
        # adding self_play.num_of_processes makes games *slower*, since workers are now
        # queueing for a saturated server rather than being throughput-limited by it).
        # Splitting the same GPU across multiple server processes parallelizes that dispatch
        # loop across more CPU cores instead of one. [] -> a single CPU-only server.
        'gpu_ids': [0,0,0,0,0,0,1,1,1,1,1,1],
        'max_batch_size': 24,   # cap on boards per network() call in the inference server
        'max_wait_ms': 5,       # longest a server waits to fill a batch before flushing partial
    },

    'self_play': {
        # Workers no longer load a network or import TensorFlow (see uttt/inference/server.py),
        # so this can go well past physical core count on a box with a real GPU behind it -
        # more concurrent workers means more simultaneous in-flight inference requests, which
        # is what actually grows batch sizes. Tune alongside 'inference' above once profiled.
        'num_of_processes': 196,
        'num_of_self_play_games_per_process': 8,
        'num_of_testing_games': 196,       # 240? candidate vs. champion gating match total, parallelized across num_of_processes
        'promotion_win_rate': 0.55,       # candidate must win >= this share of decisive (non-drawn) gating games to be promoted
        'temperature_moves': 20,       # sample proportional to visit counts for this many plies, then play greedy

        # If the self-play/gating worker pool produces no progress message at all for this
        # long, TrainingManager assumes a worker died silently (multiprocessing.Pool never
        # resolves the AsyncResult for a task whose worker process died mid-task, so nothing
        # else can tell "still working" apart from "gone forever") and forces a clean
        # restart of that phase rather than hang indefinitely. Keep this comfortably above
        # the slowest single game you've actually observed - self-play games have taken
        # 30-40+ minutes on constrained/oversubscribed hardware, and since progress is only
        # reported per finished game, it's normal for zero messages to arrive for up to
        # about one game's worth of time right after a phase starts (every worker begins
        # its first game at once).
        'stall_timeout_s': 3600,
        'max_stall_retries': 2,       # give up and raise for real after this many consecutive stalls in one phase, rather than retrying forever if the cause is systemic
    },

    'training': {
        'num_of_episodes': 100,
        'training_sample_size': 100000,   # examples drawn from the buffer to fit on this episode
        'minibatch_size': 512,          # batch_size passed to model.fit
        'training_epochs': 20,
        'training_patience': 3,
        'max_training_examples': 1000000,
        'random_symmetry_augmentation': True,  # randomly reorient each sampled example (one of the board's 8 dihedral symmetries) before every fit() call
    }
}
