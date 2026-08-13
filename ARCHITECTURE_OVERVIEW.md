# UTTT AlphaZero — Architecture Overview

## 1. What the project is

An AlphaZero-style self-play reinforcement learning system for **Ultimate Tic-Tac-Toe (UTTT)**:
a neural network (policy + value heads) guides a Monte-Carlo Tree Search, the network trains on
the results of its own self-play games, and the cycle repeats for many "episodes."

`Networks/Episode_0` through `Episode_19` (dated Sept 2022) are artifacts of an earlier run, from
before the current MCTS/training implementation — not representative of what the current code
produces.

## 2. Component map

| File | Role |
|---|---|
| [Config.py](Config.py) | Single dict of hyperparameters (MCTS, self-play, training). Source of truth for all tunable knobs — see §5. |
| [UTTTBoard.py](UTTTBoard.py) | The game engine: 9 sub-boards packed into 81-bit `x`/`o` bitboards, tracks which sub-boards are decided and which are legal to move into. Supports `make_move`/`unmake_move` (see §4) and produces the `(9,9,4)` tensor fed to the network via `get_array_representation()`. |
| [MCTS.py](MCTS.py) | The search tree. A node = a board position (visit stats, not its own board copy — see §4). Supports both a network-free random rollout mode and a network-guided PUCT-style mode. |
| [PlayerAgent.py](PlayerAgent.py) | Wraps an `MCTS` (or raw network, or pure random) behind a common `get_move`/`make_move` interface; also has the head-to-head game/match runner used for evaluation (`agent_game`, `agent_match`, `test_network_vs_mcts`, `test_raw_network_vs_random`). |
| [TrainingExample.py](TrainingExample.py) | Converts a board + MCTS visit-count policy into a training tensor triple `(board_array, search_probs, reward)`. |
| [simulate_self_play_games.py](simulate_self_play_games.py) | Plays full self-play games with the current network, producing `TrainingExample`s. This is the function farmed out to worker processes by `TrainingManager`. |
| [simulate_raw_mcts_games.py](simulate_raw_mcts_games.py) | Network-free counterpart to `simulate_self_play_games.py`: plays full games with `MCTS(network=None)` (random-rollout evaluation, uniform priors) at `config['mcts']['pretrain_mcts_depth']`, producing the same `TrainingExample`s. Never imports TensorFlow. Farmed out to worker processes by `GenerateRawMCTSExamples.py`. |
| [TrainingManager.py](TrainingManager.py) | The outer loop: on construction, resumes the replay buffer from `TrainingExamples/` if anything's there (see §2 "Starting/resuming a run"); then per episode, self-plays with the current champion, trains a *candidate* cloned from the champion, gates the candidate against the champion head-to-head and only promotes/checkpoints it if it clears a win-rate margin (§2 data flow, §4), evaluates against baselines, logs metrics to `training_log.csv`/`gating_log.csv`. |
| [GenerateRawMCTSExamples.py](GenerateRawMCTSExamples.py) | Standalone (and `Pretrain.py`-callable) pool builder: `generate_pool()` parallelizes `simulate_raw_mcts_games.py` across `mp.Pool(num_of_processes)`, saving incrementally to `PretrainExamples/pool.pkl` until it reaches `training.max_training_examples` examples. Idempotent - a no-op once the pool already meets target, so it's safe to call unconditionally on every `Pretrain.py` run. |
| [PretrainManager.py](PretrainManager.py) | Raw-MCTS bootstrap counterpart to `TrainingManager`: loads the pool `GenerateRawMCTSExamples.py` built and, per round, samples `training_sample_size` examples and trains `Network.keras` *in place* (no clone, no gating - see §2 pretraining data flow). Checkpoints every round to `Networks/Pretrain_N.keras`, logs to `pretrain_log.csv`. |
| [NetworkArchitectureTester.py](NetworkArchitectureTester.py) | Defines three candidate architectures (`denseNet`, `convNet`, `resNet`). Only `convNet()` is used by `Project.py`/`Pretrain.py`. |
| [ELO.py](ELO.py) | Elo-rating engine (`ELOManager`): random-pairing calibration loop with progress/ETA printing. Its own `__main__` round-robins baseline agents (random / rollout-MCTS at various depths) sequentially, in-process. `NetworkTournament.py` only reuses its `expected_result`/`delta_elo` static methods (its own tournament loop is parallelized separately, see below) — `ELOManager.calibrate()` itself is not wired into the main training loop. |
| [simulate_tournament_games.py](simulate_tournament_games.py) | Plays one tournament game between two agents built from lightweight specs (`{kind, name, path/depth}`). This is the function farmed out to worker processes by `NetworkTournament.run_tournament`, mirroring `simulate_self_play_games.py`/`simulate_gating_games.py`: `init_tournament_worker` (a `Pool` initializer) loads every agent once per worker, not once per game. |
| [NetworkTournament.py](NetworkTournament.py) | Standalone analysis tool: snapshots a subsampled ladder of `Networks/Episode_N.keras` checkpoints (so it never races a live training run's saves), builds specs for `ProbabilisticNetworkMCTSAgent`s from them plus network-free `RolloutMCTSAgent` baselines at several depths, and runs one combined Elo round-robin — parallelized across a `multiprocessing.Pool` the same way `TrainingManager` parallelizes self-play/gating (worker pool + a shared `Manager().Queue()` for progress; `run_tournament`'s main process is the only place Elo ratings are read or mutated, applying each update as results stream back in non-deterministic completion order). Writes ranked standings + an Elo-trajectory CSV. Answers "is the network better than plain search" as well as "which episode is best." Not wired into the training loop. |
| [TTTBoard.py](TTTBoard.py) | Plain 3×3 tic-tac-toe board. Not used by the training/play pipeline (`UTTTBoard` is fully self-contained); only referenced from the stale `Project.ipynb`. |
| [Project.py](Project.py) | Entry point: build a fresh `convNet` and save it as `Network.keras` only if that file doesn't already exist (otherwise a restart resumes the existing champion instead of discarding it), then run `TrainingManager().train()`. This is the actively maintained entry point — `Project.ipynb` / `.ipynb_checkpoints` are stale and out of sync with it. |
| [Pretrain.py](Pretrain.py) | Alternative, optional entry point that runs *before* `Project.py`: same `convNet`-if-missing bootstrap, then unconditionally calls `GenerateRawMCTSExamples.generate_pool()` followed by `PretrainManager().pretrain()` - one fire-and-forget command that builds/resumes the raw-MCTS pool and trains against it with no self-play or gating (§2 pretraining data flow). Meant as a one-time bootstrap before switching to `Project.py`, not a standing replacement for it. |

### Starting/resuming a run

Both the network and the replay buffer persist across `python Project.py` invocations, not just
across episodes within one run:

- **Network**: `Project.py` only builds+saves a fresh `convNet` if `Network.keras` doesn't already
  exist. If it does, `TrainingManager.__init__` loads it as the starting champion, so restarting
  the process resumes training rather than discarding it.
- **Replay buffer**: `TrainingManager.load_latest_training_examples()` looks in `TrainingExamples/`
  for the most-recently-modified file (not the highest `Episode_N` — numbering restarts at 0 every
  run, so a stale file from an earlier run can have a higher N than the real latest, the same
  footgun `NetworkTournament.py` already accounts for with `Networks/Episode_N.keras`), loads it as
  the starting buffer, renames it to a single fixed backup name `TrainingExamples/_resumed_from.pkl`,
  and deletes every other file so stale runs' episode files never coexist ambiguously with the new
  run's own numbering. Each saved file already holds the *entire* buffer as of that episode (not
  just that episode's new examples — see below), so the single most recent file already is
  essentially the full prior buffer; there's nothing to merge across files.

### Data flow, one training episode

```
Project.py / TrainingManager.train()
  │
  ├─ mp.Pool(num_of_processes): simulate_self_play_games() per worker, playing
  │  against the current champion (whatever Network.keras currently holds)
  │     (each worker loads Network.keras, seeds its own RNG from os.getpid(),
  │      caps itself to 1 TF thread to avoid oversubscription across workers)
  │     └─ per game: MCTS.search(add_root_noise=True) → sample move
  │                  (∝ visit counts for the first temperature_moves plies,
  │                   greedy/argmax after) → TrainingExample
  │                  … until game over → back-fill reward into each example
  │
  ├─ accumulate into TrainingManager.training_examples (capped at max_training_examples, FIFO trim)
  │
  ├─ clone_network(champion) → candidate; sample training_sample_size examples →
  │  one shuffled model.fit() on the candidate only (validation_split, EarlyStopping
  │  on val_loss) — the champion itself is never mutated. Metrics logged to
  │  training_log.csv.
  │
  ├─ gate: agent_match(candidate, champion, num_of_testing_games), both sides
  │  wrapped as ProbabilisticNetworkMCTSAgent (samples ∝ visit counts, so repeated
  │  games actually differ - the deterministic NetworkMCTSAgent would replay the
  │  same game every time in a fixed color order). Candidate must win at least
  │  self_play.promotion_win_rate (default 0.55) of decisive (non-drawn) games to
  │  be promoted - a bare wins > losses majority isn't enough, since with
  │  num_of_testing_games in the tens a 1-2 game margin is statistically
  │  indistinguishable from a coin flip. Falling short keeps the old champion.
  │  Outcome logged to gating_log.csv.
  │
  ├─ if promoted: candidate becomes the new champion → save Networks/Episode_N.keras
  │  and Network.keras. If rejected: no checkpoint is written this episode, so
  │  Networks/ ends up sparse (Episode_0, Episode_4, Episode_17, …) rather than dense.
  ├─ test_network_vs_mcts() / test_raw_network_vs_random()   (printed only, against
  │  whichever network Network.keras currently holds)
  └─ pickle training_examples to TrainingExamples/Episode_N
```

### Data flow, raw-MCTS pretraining (optional, before self-play)

```
Pretrain.py
  │
  ├─ build convNet + save Network.keras, only if it doesn't already exist
  │
  ├─ GenerateRawMCTSExamples.generate_pool() - no-op if PretrainExamples/pool.pkl
  │  already holds >= max_training_examples, otherwise loops:
  │     mp.Pool(num_of_processes): simulate_raw_mcts_games() per worker
  │        (MCTS(network=None) - random rollout evaluation, uniform priors;
  │         same temperature_moves schedule as self-play, no Dirichlet noise,
  │         never touches TensorFlow) → save pool to disk after every batch
  │  until the pool reaches max_training_examples, then trims to exactly that size
  │
  └─ PretrainManager.pretrain() - per round (training.num_of_episodes rounds):
       sample training_sample_size examples from the (now-static) pool →
       model.fit() on Network.keras directly - no clone_network(), no gating,
       every round is kept unconditionally →
       save Networks/Pretrain_N.keras and Network.keras →
       log metrics to pretrain_log.csv
```

Unlike self-play, the pool is generated once and reused across all rounds - it's not
regenerated per round, and nothing in this pipeline gates a round against a prior one.
This is meant as a one-time bootstrap: raw MCTS at a fixed depth doesn't get stronger
no matter how many times it's re-run, so once `pretrain_log.csv` / `NetworkTournament.py`
results plateau, the intended next step is `python Project.py`, not more pretraining
rounds. `Networks/Pretrain_N.keras` numbering restarts at 0 every `Pretrain.py` run,
the same footgun `Episode_N` already has (see "Starting/resuming a run" above) - a second
pretraining run (e.g. after manually regenerating the pool) can silently overwrite the
first run's checkpoint files, though `Network.keras` itself always holds the latest
weights regardless.

## 3. MCTS node lifecycle

Each `MCTS` node holds `t` (total backpropagated value), `n` (visit count), `children`, and
`search_probs` (network priors over legal moves, mixed with Dirichlet noise at the root during
self-play — see §4). The first visit to a node evaluates it directly (network or random rollout)
and backpropagates; only the *second* visit actually materializes `children`. This is a lazy-
expansion scheme, so `sum(child.n for child in node.children) == node.n - 1` is an expected
invariant, not a bug.

`MCTS.make_move()` is the tree-shift operation used between plies: given a move, it looks for a
matching existing child (reusing its accumulated search stats) or constructs a fresh node,
returning whichever becomes the new root for the next ply.

## 4. Board representation & search performance design

A few design choices exist specifically to keep self-play fast, since MCTS spends most of its time
evaluating leaves:

- **Single shared board via make/unmake, not per-node copies.** Only the root of an `MCTS` tree
  owns a real `UTTTBoard`. Descending into a child during search is `board.make_move(child.move)`
  → recurse → `board.unmake_move()`, threaded explicitly through `MCTS_iteration`/`random_rollout`/
  `check_network`/`make_children_list` as a `board` parameter. `UTTTBoard.make_move()` pushes a
  small undo record (the move, whose turn it was, and the three small scalar/list fields it's
  about to overwrite) onto an internal `_history` stack; `unmake_move()` pops and restores it
  exactly. No board is ever copied during search.
- **Direct model calls, not `.predict()`.** Leaf evaluation calls `self.network(board_arrays,
  training=False)` rather than `model.predict(...)`, avoiding `.predict()`'s per-call overhead
  (dataset/callback machinery) that dominates when called once per MCTS leaf.
- **Vectorized board-array encoding.** `UTTTBoard.symbol_array_representation()` builds the
  `(9,9)` per-symbol plane with a single `np.unpackbits()` call (via `int.to_bytes()`, since the
  81-bit bitboard doesn't fit in an `int64`) followed by a reshape/transpose/flip, instead of a
  Python loop over subboards doing string-based bit parsing.
- **Root Dirichlet noise, self-play only.** `MCTS.search(add_root_noise=True)` mixes
  `dirichlet_epsilon` worth of `Dirichlet(dirichlet_alpha)` noise into the root's priors right
  after its first evaluation, before any UCB descent uses them. Only self-play passes
  `add_root_noise=True`; evaluation/head-to-head play (`NetworkMCTSAgent`, `test_network_vs_mcts`,
  etc.) does not, so noise never affects real play strength measurement.
- **Temperature annealing.** `simulate_self_play_games()` samples moves proportional to visit
  counts (`p=mcts.pi`) for the first `temperature_moves` plies of a game, then plays greedily
  (`argmax`) afterward.
- **Per-worker RNG seeding + thread caps.** Each self-play worker process seeds `random`/`numpy`
  from its own PID and caps itself to a single TF thread
  (`set_intra_op_parallelism_threads(1)`/`set_inter_op_parallelism_threads(1)`), so
  `num_of_processes` workers running concurrently don't each spin up a full per-core thread pool
  and oversubscribe the CPU.
- **Consolidated self-play progress.** Workers report per-game completions through a shared
  `multiprocessing.Manager().Queue()` (built once in `TrainingManager.__init__`, reused every
  episode) instead of each process writing to stdout independently; `TrainingManager.run_self_play()`
  drains it into one ETA'd progress stream. `train_on_examples()` also runs `model.fit(verbose=2)`
  (one line/epoch) rather than silent, and `PlayerAgent.agent_match()` prints one line per game with
  a running tally and ETA — all three exist so a long-running phase never looks hung.

## 5. Config knobs (`Config.py`)

- `mcts.exploration_parameter` — PUCT exploration constant.
- `mcts.search_depth` — MCTS iterations per move.
- `mcts.dirichlet_alpha` / `mcts.dirichlet_epsilon` — root noise shape/weight during self-play (§4).
- `mcts.pretrain_mcts_depth` — MCTS iterations per move for network-free pretraining pool generation
  (`GenerateRawMCTSExamples.py`), independent of `mcts.search_depth`.
- `self_play.num_of_processes` — parallel self-play worker processes per episode; also the worker
  count `GenerateRawMCTSExamples.py` uses per generation batch.
- `self_play.num_of_self_play_games_per_process` — games each worker plays before returning; also
  reused by `simulate_raw_mcts_games.py` for pretraining generation batches.
- `self_play.num_of_testing_games` — games per champion-vs-candidate gating match each episode
  (§2), and separately per evaluation match in `test_network_vs_mcts()`/`test_raw_network_vs_random()`.
- `self_play.promotion_win_rate` — minimum share of decisive gating games the candidate must win to
  be promoted (§2, §4).
- `self_play.temperature_moves` — plies played proportional-to-visits before switching to greedy
  (§4); same schedule reused by pretraining generation.
- `training.num_of_episodes` — outer training loop length; reused as the round count for
  `PretrainManager.pretrain()`.
- `training.training_sample_size` — examples sampled from the replay buffer to fit on each episode.
- `training.minibatch_size` — `batch_size` passed to `model.fit`.
- `training.training_epochs` / `training.training_patience` — max epochs and `EarlyStopping`
  patience (monitoring `val_loss`) for that one `fit()` call.
- `training.max_training_examples` — replay buffer cap (oldest examples trimmed first).

## 6. Known gaps / not yet implemented

- **No symmetry data augmentation.** UTTT has 8 dihedral board symmetries; training data isn't
  multiplied using them, which is otherwise a cheap way to get more mileage out of self-play data.
- **No batched leaf evaluation (virtual loss).** Each MCTS leaf still costs one model call; real
  AlphaZero implementations batch several pending leaf evaluations into one forward pass.
- **`test_network_vs_mcts()` / `test_raw_network_vs_random()` are print-only.** Unlike the
  champion-vs-candidate gating match (logged to `gating_log.csv`) and training metrics (logged to
  `training_log.csv`), these two baseline evaluations only `print()` a W/D/L line per episode —
  nothing is written to a file.
- **`ELO.py`/`NetworkTournament.py`** are standalone analysis tools, not wired into the training loop.
- **Pretraining pool has no versioning/lineage.** Regenerating `PretrainExamples/pool.pkl` (e.g.
  `rm PretrainExamples/pool.pkl && python Pretrain.py`) is a manual, destructive replace — there's
  no way to compare, mix, or trace which `Networks/Pretrain_N.keras` checkpoint was trained against
  which pool generation.
- **`Project.ipynb` / `.ipynb_checkpoints`** are stale relative to `Project.py` and reference
  removed patterns (e.g. `TTTBoard`, `disable_eager_execution()`); treat `Project.py` as the
  source of truth.
