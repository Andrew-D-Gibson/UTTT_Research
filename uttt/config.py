config = {
    # Bump this by hand any time you change a value below. It gets snapshotted once to
    # data/logs/config_history/config_v{version}.json and stamped on every training_log.csv/
    # gating_log.csv row, so a given episode's exact hyperparameters can always be
    # recovered later, even if config.py has since moved on (see uttt/run_logging.py).
    'version': 1,

    'mcts': {
        'exploration_parameter': 1.5,
        'search_depth': 512,           
        'dirichlet_alpha': 1.2,       # root exploration noise (AlphaZero-style)
        'dirichlet_epsilon': 0.25,    # weight of noise vs. network prior at the root
    },

    'self_play': {
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
