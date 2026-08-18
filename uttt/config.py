config = {
    # Bump this by hand any time you change a value below. It gets snapshotted once to
    # data/logs/config_history/config_v{version}.json and stamped on every training_log.csv/
    # gating_log.csv row, so a given episode's exact hyperparameters can always be
    # recovered later, even if config.py has since moved on (see uttt/run_logging.py).
    'version': 2,

    'network': {
        'architecture': 'hierarchicalResNet',   # 'hierarchicalResNet' or 'convNet' - see uttt/network/architectures.py's build_network()
    },

    'mcts': {
        'exploration_parameter': 1.5,
        'search_depth': 512,           
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
        'gpu_ids': [],
        'max_batch_size': 64,   # cap on boards per network() call in the inference server
        'max_wait_ms': 5,       # longest a server waits to fill a batch before flushing partial
    },

    'self_play': {
        # Workers no longer load a network or import TensorFlow (see uttt/inference/server.py),
        # so this can go well past physical core count on a box with a real GPU behind it -
        # more concurrent workers means more simultaneous in-flight inference requests, which
        # is what actually grows batch sizes. Tune alongside 'inference' above once profiled.
        'num_of_processes': 12,
        'num_of_self_play_games_per_process': 40,
        'num_of_testing_games': 60,       # 240? candidate vs. champion gating match total, parallelized across num_of_processes
        'promotion_win_rate': 0.55,       # candidate must win >= this share of decisive (non-drawn) gating games to be promoted
        'temperature_moves': 20,       # sample proportional to visit counts for this many plies, then play greedy
    },

    'training': {
        'num_of_episodes': 100,
        'training_sample_size': 40000,   # examples drawn from the buffer to fit on this episode
        'minibatch_size': 512,          # batch_size passed to model.fit
        'training_epochs': 10,
        'training_patience': 3,
        'max_training_examples': 400000,
        'random_symmetry_augmentation': True,  # randomly reorient each sampled example (one of the board's 8 dihedral symmetries) before every fit() call
    }
}
