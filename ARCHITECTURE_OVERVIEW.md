# UTTT AlphaZero — Architecture Overview

## 1. What the project is

An AlphaZero-style self-play reinforcement learning system for **Ultimate Tic-Tac-Toe (UTTT)**:
a neural network (policy + value heads) guides a Monte-Carlo Tree Search, the network trains on
the results of its own self-play games, and the cycle repeats for many "episodes."

`data/Networks/Episode_0` through `Episode_19` (dated Sept 2022) are artifacts of an earlier run,
from before the current MCTS/training implementation — not representative of what the current code
produces.

## 2. Component map

| File | Role |
|---|---|
| [uttt/config.py](uttt/config.py) | Single dict of hyperparameters (MCTS, self-play, training). Source of truth for all tunable knobs — see §5. |
| [uttt/board/uttt_board.py](uttt/board/uttt_board.py) | The game engine: 9 sub-boards packed into 81-bit `x`/`o` bitboards, tracks which sub-boards are decided and which are legal to move into. Supports `make_move`/`unmake_move` (see §4) and produces the `(9,9,4)` tensor fed to the network via `get_array_representation()`. |
| [uttt/search/mcts.py](uttt/search/mcts.py) | The search tree. A node = a board position (visit stats, not its own board copy — see §4). Supports both a network-free random rollout mode and a network-guided PUCT-style mode. |
| [uttt/player/agent.py](uttt/player/agent.py) | Wraps an `MCTS` (or pure random) behind a common `get_move`/`make_move` interface; also has the head-to-head game/match runner used for evaluation (`agent_game`, `agent_match`). |
| [uttt/training/example.py](uttt/training/example.py) | Converts a board + MCTS visit-count policy into a training tensor triple `(board_array, search_probs, reward)`. |
| [uttt/simulation/self_play.py](uttt/simulation/self_play.py) | Plays full self-play games with the current network, producing `TrainingExample`s. This is the function farmed out to worker processes by `TrainingManager`. |
| [uttt/training/manager.py](uttt/training/manager.py) | The outer loop: on construction, resumes the replay buffer from `data/TrainingExamples/` if anything's there (see §2 "Starting/resuming a run"); then per episode, self-plays with the current champion, trains a *candidate* cloned from the champion, gates the candidate against the champion head-to-head and only promotes/checkpoints it if it clears a win-rate margin (§2 data flow, §4), evaluates against baselines, logs metrics to `data/logs/training_log.csv`/`data/logs/gating_log.csv`. |
| [uttt/network/architectures.py](uttt/network/architectures.py) | Defines the network architecture (`convNet`, policy+value heads) used by `project.py`. |
| [uttt/evaluation/elo.py](uttt/evaluation/elo.py) | `ELOManager`: just the `expected_result`/`delta_elo` static Elo-math helpers, reused by `uttt/evaluation/network_tournament.py`'s tournament loop (which is parallelized separately, see below). |
| [uttt/simulation/tournament.py](uttt/simulation/tournament.py) | Plays one tournament game between two agents built from lightweight specs (`{kind, name, path/depth}`). This is the function farmed out to worker processes by `run_tournament` in `uttt/evaluation/network_tournament.py`, mirroring `uttt/simulation/self_play.py`/`uttt/simulation/gating.py`: `init_tournament_worker` (a `Pool` initializer) loads every agent once per worker, not once per game. |
| [uttt/evaluation/network_tournament.py](uttt/evaluation/network_tournament.py) | Standalone analysis tool: snapshots a subsampled ladder of `data/Networks/Episode_N.keras` checkpoints (so it never races a live training run's saves), builds specs for `ProbabilisticNetworkMCTSAgent`s from them plus network-free `RolloutMCTSAgent` baselines at several depths, and runs one combined Elo round-robin — parallelized across a `multiprocessing.Pool` the same way `TrainingManager` parallelizes self-play/gating (worker pool + a shared `Manager().Queue()` for progress; `run_tournament`'s main process is the only place Elo ratings are read or mutated, applying each update as results stream back in non-deterministic completion order). Writes ranked standings + an Elo-trajectory CSV to `data/logs/`. Answers "is the network better than plain search" as well as "which episode is best." Not wired into the training loop. Invoked via the root [network_tournament.py](network_tournament.py) launcher's `run_default()`. |
| [project.py](project.py) | Entry point: build a fresh `convNet` and save it as `data/Network.keras` only if that file doesn't already exist (otherwise a restart resumes the existing champion instead of discarding it), then run `TrainingManager().train()`. |

### Starting/resuming a run

Both the network and the replay buffer persist across `python project.py` invocations, not just
across episodes within one run:

- **Network**: `project.py` only builds+saves a fresh `convNet` if `data/Network.keras` doesn't
  already exist. If it does, `TrainingManager.__init__` loads it as the starting champion, so
  restarting the process resumes training rather than discarding it.
- **Replay buffer**: `TrainingManager.load_latest_training_examples()` loads the starting buffer
  from one fixed-name file, `data/TrainingExamples/current_training_examples.pkl`, which always
  holds the entire (post-trim) buffer as of the last episode that ran. `data/TrainingExamples/
  Episode_N` files are a separate, permanent per-episode audit trail of just that episode's fresh
  self-play examples (not the whole buffer, and not trimmed) — they're written every episode but
  never read back on resume, so unlike `data/Networks/Episode_N.keras` their restart-at-0 numbering
  across runs is harmless (nothing depends on `N` being globally increasing).

### Data flow, one training episode

```
project.py / TrainingManager.train()
  │
  ├─ start_inference_servers(): one run_inference_server process per config.inference.gpu_id,
  │  each loading its own copy of data/Network.keras and batching leaf-evaluation requests
  │  from all self-play workers (uttt/inference/server.py) - see §4a.
  │
  ├─ mp.Pool(num_of_processes, initializer=init_self_play_worker, initargs=(request_queue,)):
  │  simulate_self_play_games() per worker, playing against the current champion
  │     (workers never load a network or import TensorFlow themselves - each just seeds
  │      its own RNG from os.getpid() and holds an InferenceClient that submits leaf
  │      boards to request_queue and blocks for a batched response)
  │     └─ per game: MCTS.search(add_root_noise=True) → sample move
  │                  (∝ visit counts for the first temperature_moves plies,
  │                   greedy/argmax after) → TrainingExample
  │                  … until game over → back-fill reward into each example
  │
  ├─ accumulate into TrainingManager.training_examples (capped at max_training_examples, FIFO trim)
  │
  ├─ clone_network(champion) → candidate; sample training_sample_size examples →
  │  one shuffled model.fit() on the candidate only (validation_split, EarlyStopping
  │  on val_loss) — the champion itself is never mutated.
  │
  ├─ gate: agent_match(candidate, champion, num_of_testing_games), both sides
  │  wrapped as ProbabilisticNetworkMCTSAgent (samples ∝ visit counts, so repeated
  │  games actually differ - a deterministic argmax agent would replay the
  │  same game every time in a fixed color order). Candidate must win at least
  │  self_play.promotion_win_rate (default 0.55) of decisive (non-drawn) games to
  │  be promoted - a bare wins > losses majority isn't enough, since with
  │  num_of_testing_games in the tens a 1-2 game margin is statistically
  │  indistinguishable from a coin flip. Falling short keeps the old champion.
  │  Outcome logged to data/logs/gating_log.csv.
  │
  ├─ if promoted: candidate becomes the new champion → save data/Networks/Episode_N.keras
  │  and data/Network.keras. If rejected: no checkpoint is written this episode, so
  │  data/Networks/ ends up sparse (Episode_0, Episode_4, Episode_17, …) rather than dense.
  │
  ├─ pickle this episode's fresh examples to data/TrainingExamples/Episode_N (audit trail,
  │  never trimmed/re-read) and the full trimmed buffer to
  │  data/TrainingExamples/current_training_examples.pkl (what actually gets resumed from)
  │
  └─ one row appended to data/logs/training_log.csv: self-play/training/episode durations,
     self-play game/ply counts, buffer size, and the train_on_examples() metrics - stamped
     with timestamp and config_version (see §7 "Auditability & logging" below)
```

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
  `add_root_noise=True`; evaluation/head-to-head play (gating, tournaments, etc.) does not, so
  noise never affects real play strength measurement.
- **Temperature annealing.** `simulate_self_play_games()` samples moves proportional to visit
  counts (`p=mcts.pi`) for the first `temperature_moves` plies of a game, then plays greedily
  (`argmax`) afterward.
- **Per-worker RNG seeding.** Each self-play worker process seeds `random`/`numpy` from its own PID
  (`uttt/worker.py`'s `seed_worker_rng()`) so parallel games don't end up correlated. Workers no
  longer import TensorFlow at all (see §4a), so there's no per-process TF thread pool to cap
  anymore - `configure_cpu_worker()`'s thread-capping half only still runs inside
  `run_inference_server` and `uttt/simulation/tournament.py`'s worker init (a separate,
  not-batched tool - see §6).
- **Consolidated self-play progress.** Workers report per-game completions through a shared
  `multiprocessing.Manager().Queue()` (built once in `TrainingManager.__init__`, reused every
  episode) instead of each process writing to stdout independently; `TrainingManager.run_self_play()`
  drains it into one ETA'd progress stream. `train_on_examples()` also runs `model.fit(verbose=2)`
  (one line/epoch) rather than silent, and `PlayerAgent.agent_match()` prints one line per game with
  a running tally and ETA — all three exist so a long-running phase never looks hung.

### 4a. Batched inference across workers (`uttt/inference/server.py`)

Self-play/gating workers no longer load a network themselves. Each holds an `InferenceClient`
that implements the same `__call__(board_arrays, training=False)` interface `MCTS.check_network`
already calls, so `MCTS`'s tree-walk logic is completely unaware whether it's talking to a real
in-process Keras model (as `uttt/interface.py` and `network_tournament.py` still use directly) or
this stand-in:

```
worker (InferenceClient)                     inference server (run_inference_server)
  __call__(board) ─┐                           owns the one loaded network for this GPU
                    ├─ put (board, response_conn) on a shared request_queue
                    └─ blocks on response_conn.recv() ──┐
                                                          ├─ drains request_queue up to
                                                          │  inference.max_batch_size boards or
                                                          │  inference.max_wait_ms, whichever first
                                                          ├─ one network(stacked_boards) call
                                                          └─ sends each result back down its
                                                             requester's own response_conn
```

Batching comes from many *different* games' leaf requests landing at the server around the same
time (purely because many worker processes are mid-search simultaneously) - MCTS's own recursive
search (`MCTS_iteration`) is untouched, still fully synchronous per game, no virtual loss. One
server process per `config['inference']['gpu_ids']` entry (`[]` → a single CPU-only server).
`TrainingManager.start_inference_server(s)`/`stop_inference_servers()` own the server processes'
lifecycle, one set per `run_self_play()`/`run_gating()` call (`run_gating` runs two - candidate and
champion are different weights, so each needs its own server rather than sharing one).

The request queue is passed to `Pool` workers via `initializer`/`initargs`
(`init_self_play_worker`/`init_gating_worker`), not as a normal `apply_async` argument - a raw
`multiprocessing.Queue` can only be inherited by a worker at process-creation time; pickling one
through `Pool`'s per-task dispatch queue raises `RuntimeError: Queue objects should only be shared
between processes through inheritance`. This is the same reason `uttt/simulation/tournament.py`
already builds its agents via a `Pool(initializer=...)` rather than per-call arguments.

## 5. Config knobs (`uttt/config.py`)

- `mcts.exploration_parameter` — PUCT exploration constant.
- `mcts.search_depth` — MCTS iterations per move.
- `mcts.dirichlet_alpha` / `mcts.dirichlet_epsilon` — root noise shape/weight during self-play (§4).
- `inference.gpu_ids` — one inference server process per id (`[]` → single CPU-only server); §4a.
- `inference.max_batch_size` / `inference.max_wait_ms` — batching policy for each inference server
  (§4a): boards per `network()` call, and how long to wait to fill a batch before flushing partial.
- `self_play.num_of_processes` — parallel self-play worker processes per episode. No longer bounded
  by physical core count the way it used to be (workers don't load a network or import TensorFlow
  anymore), and is also the main lever for inference batch size - more concurrent workers means
  more simultaneous in-flight requests hitting each server.
- `self_play.num_of_self_play_games_per_process` — games each worker plays before returning.
- `self_play.num_of_testing_games` — games per champion-vs-candidate gating match each episode (§2).
- `self_play.promotion_win_rate` — minimum share of decisive gating games the candidate must win to
  be promoted (§2, §4).
- `self_play.temperature_moves` — plies played proportional-to-visits before switching to greedy
  (§4).
- `training.num_of_episodes` — outer training loop length.
- `training.training_sample_size` — examples sampled from the replay buffer to fit on each episode.
- `training.minibatch_size` — `batch_size` passed to `model.fit`.
- `training.training_epochs` / `training.training_patience` — max epochs and `EarlyStopping`
  patience (monitoring `val_loss`) for that one `fit()` call.
- `training.max_training_examples` — replay buffer cap (oldest examples trimmed first).

## 6. Known gaps / not yet implemented

- **No symmetry data augmentation.** UTTT has 8 dihedral board symmetries; training data isn't
  multiplied using them, which is otherwise a cheap way to get more mileage out of self-play data.
- **No in-tree virtual loss.** Leaf evaluations are now batched *across* concurrently-running
  self-play/gating games (§4a), but a single MCTS search still evaluates one leaf, fully
  backpropagates, then starts the next iteration - it never has multiple leaves "pending" within
  the same tree the way virtual-loss AlphaZero implementations do. Batch size is therefore capped
  by how many games happen to be mid-search at once, not by search depth.
- **`uttt/evaluation/network_tournament.py`** is a standalone analysis tool, not wired into the
  training loop.

## 7. Auditability & logging

`uttt/run_logging.py` holds two pieces of infrastructure used by a long training run, orthogonal
to each other:

- **`snapshot_config()`** (called once, in `TrainingManager.__init__`) writes the live `config`
  dict to `data/logs/config_history/config_v{config['version']}.json` the first time that version
  number is seen — a no-op on later calls with the same version, so it's safe to call every run.
  `version` is hand-bumped in `uttt/config.py`, not derived from anything (not a hash, not git) —
  if you edit a hyperparameter without bumping it, the next `snapshot_config()` call detects the
  mismatch against the existing snapshot and prints a warning, but does not overwrite the snapshot
  or block the run. Every row of `data/logs/training_log.csv` and `data/logs/gating_log.csv` is
  stamped with `config_version`, so an episode's exact hyperparameters are recoverable later via
  that file even after `config.py` has since changed further.
- **`start_console_log()`** (called once, in `project.py`) tees `sys.stdout`/`sys.stderr` through
  to a new file per process under `data/logs/console/run_<start-timestamp>.log`, in addition to the
  terminal — nothing printed anywhere in the process (self-play/gating progress, `print_settings()`,
  Keras's own `fit()` output) is lost if the terminal session ends. Every row of `training_log.csv`/
  `gating_log.csv` also carries a `timestamp`, which is how you'd locate the matching console log
  for a given episode (e.g. to find a stack trace).

Both CSVs' `episode` column is the join key back to `data/TrainingExamples/Episode_N` and (for
promoted episodes) `data/Networks/Episode_N.keras`.
