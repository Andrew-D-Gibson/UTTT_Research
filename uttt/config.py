config = {
    'mcts': {
        'exploration_parameter': 4,
        'search_depth': 512,           
        'dirichlet_alpha': 0.3,       # root exploration noise (AlphaZero-style)
        'dirichlet_epsilon': 0.25,    # weight of noise vs. network prior at the root
        'pretrain_mcts_depth': 4096,  # iterations for raw (network-free) MCTS during pool generation - see uttt/training/generate_raw_pool.py
    },

    'self_play': {
        'num_of_processes': 12,                     
        'num_of_self_play_games_per_process': 20, 
        'num_of_testing_games': 60,       # candidate vs. champion gating match total, parallelized across num_of_processes
        'promotion_win_rate': 0.55,       # candidate must win >= this share of decisive (non-drawn) gating games to be promoted
        'num_of_baseline_games': 10,      # sequential baseline evals (test_network_vs_mcts, test_raw_network_vs_random)
        'temperature_moves': 10,       # sample proportional to visit counts for this many plies, then play greedy
    },

    'training': {
        'num_of_episodes': 100,          
        'training_sample_size': 40000,   # examples drawn from the buffer to fit on this episode
        'minibatch_size': 512,          # batch_size passed to model.fit
        'training_epochs': 50,          
        'training_patience': 3,
        'max_training_examples': 400000,
    }
}
