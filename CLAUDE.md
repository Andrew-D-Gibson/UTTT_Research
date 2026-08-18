# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Environment: conda env named `UTTT` (`conda activate UTTT`). Dependencies are `numpy`,
`tensorflow>=2.9.0`, and `pygame` (`requirements.txt`); the installed environment on this machine uses Keras 3
(bundled with TF 2.21), which matters because the code must stay in normal eager-execution style —
Keras 3's save/load format is incompatible with `tf.compat.v1.disable_eager_execution()`.

All entry points are thin **launchers at the repo root** (`project.py`,
`network_tournament.py`, `interface.py`) that import the real logic from the
`uttt/` package below. Run them from the repo root — the multiprocessing workers are
farmed out as top-level functions (e.g. `uttt.simulation.self_play.simulate_self_play_games`)
so they must stay picklable by qualified name; the launchers don't change the working
directory. There's no CLI, edit `uttt/config.py` to change run scale.

- **Run training**: `python project.py` — resumes from `data/Network.keras` if it already
  exists (only builds a fresh randomly-initialized `convNet` if it doesn't), then runs
  `TrainingManager().train()` for `config['training']['num_of_episodes']` episodes. Each
  episode only promotes/checkpoints a newly-trained candidate if it beats the current
  champion head-to-head, so `data/Networks/Episode_N.keras` is written sparsely, not every
  episode (see Architecture below).
- **No test suite, linter, or build step exists.** There's no `pytest`/`unittest` usage, no lint
  config. `python project.py` running one full episode end-to-end (self-play → training → gating →
  eval matches all completing without error) is the closest thing to an integration test.
- **Network tournament tool** (separate from the training loop): `python network_tournament.py`
  snapshots a subsampled ladder of `data/Networks/Episode_N.keras` checkpoints plus raw-MCTS baselines at
  several depths and runs an Elo round-robin (via `uttt/evaluation/elo.py`) — checks whether the network actually
  beats plain search, not just whether validation loss is dropping.
- **Pygame GUI** (human vs network, or spectator): `python interface.py human --network data/Network.keras`
   / `python interface.py spectate --network1 ... --network2 ...`.

## Architecture

See [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) for the full component map, per-episode
data flow, and current design details — read it before making non-trivial changes. The load-bearing
patterns that span multiple files and are easy to miss from reading any single file:

- **`uttt/config.py`'s `config` dict is the single source of hyperparameters**, imported via
  `from uttt.config import config` into nearly every other module (`uttt/search/mcts.py`,
  `uttt/training/manager.py`, `uttt/simulation/self_play.py`, `uttt/player/agent.py`, ...). There's
  no per-module configuration — changing search depth, process count, or training schedule always
  means editing `uttt/config.py`.
- **Self-play/gating workers don't hold a network at all — inference is batched through a separate
  server process.** `TrainingManager.start_inference_server(s)` spins up one `run_inference_server`
  process per `config['inference']['gpu_ids']` entry (`uttt/inference/server.py`), each loading its
  own copy of the network and owning a `multiprocessing.Queue` that self-play/gating workers submit
  leaf-board requests to; the server batches whatever's arrived (up to `max_batch_size` boards or
  `max_wait_ms`, whichever first) into one `network()` call and routes each result back down the
  requester's own `Pipe`. Workers never import TensorFlow — each just holds an `InferenceClient`
  (same `__call__(board_arrays, training=False)` interface as a real model, so `MCTS.check_network`
  can't tell them apart). The request queue must be handed to `Pool` workers via
  `initializer`/`initargs` (`init_self_play_worker`/`init_gating_worker`), not a normal
  `apply_async` argument — a raw `Queue` can only be inherited at process-creation time, not pickled
  through `Pool`'s per-task dispatch queue. `run_gating` runs two separate servers (candidate and
  champion are different weights). See `ARCHITECTURE_OVERVIEW.md` §4a for the full design. Changes
  to how the network is saved/loaded (e.g. checkpoint file extension, format) affect
  `uttt/training/manager.py` (the writer), `uttt/inference/server.py` (the only reader during
  self-play/gating), and `uttt/player/agent.py`/`uttt/interface.py` (which still load a network
  in-process for single interactive/evaluation games) — Keras 3 requires an explicit `.keras`/`.h5`
  extension on every save path.
- **`MCTS` nodes don't own their own board.** Only the root of an `MCTS` tree holds a real
  `UTTTBoard`; every recursive call during search (`MCTS_iteration`, `random_rollout`,
  `check_network`, `make_children_list`) takes that single shared board as an explicit parameter
  and must pair `board.make_move(...)` with `board.unmake_move()` around any recursive descent into
  a child. Breaking that make/unmake balance (e.g. adding a new recursive path, an early return, or
  an exception between the two calls) silently corrupts the board for every subsequent search — this
  is the single most fragile invariant in the codebase and spans `uttt/search/mcts.py` and
  `uttt/board/uttt_board.py` together.
- **Training gates candidates against the champion; it doesn't always save.**
  `TrainingManager.train()` clones the champion into a *candidate* (`clone_network()` — the
  champion itself is never mutated), trains only the clone, then plays `num_of_testing_games` of
  candidate vs. champion (`agent_match` with `ProbabilisticNetworkMCTSAgent` on both sides — not
  a deterministic argmax agent, which would replay identical games in a fixed color order and add
  no signal). The candidate is promoted (and only then does `data/Networks/Episode_N.keras`/
  `data/Network.keras` get written) only if it wins at least `self_play.promotion_win_rate` (default
  0.55) of *decisive* (non-drawn) gating games — a bare `wins > losses` majority isn't enough, since
  with `num_of_testing_games` in the tens a 1-2 game margin is statistically indistinguishable from
  a coin flip. A result under that threshold keeps the old champion and skips the checkpoint, so
  `data/Networks/` ends up sparse (`Episode_0`, `Episode_4`, `Episode_17`, ...), not dense. Outcomes
  log to `data/logs/gating_log.csv`; training metrics log to `data/logs/training_log.csv` — both
  rows are stamped with `timestamp` and `config_version` (see below).
- **`config['version']` is a hand-bumped audit key, not a code version.** `uttt/run_logging.py`'s
  `snapshot_config()` (called once in `TrainingManager.__init__`) writes the live `config` dict to
  `data/logs/config_history/config_v{version}.json` the first time that version number is seen, and
  every row in `training_log.csv`/`gating_log.csv` is stamped with `config_version` — so an episode's
  exact hyperparameters are recoverable later even after `uttt/config.py` has since changed. Nothing
  bumps `version` automatically: editing `config.py` without incrementing it makes `snapshot_config()`
  print a warning (existing snapshot no longer matches) but does **not** overwrite the old snapshot or
  block the run, so a forgotten bump silently means later episodes are mis-attributed to the old
  version's snapshot until you notice the warning. `project.py` separately calls `start_console_log()`
  to tee `sys.stdout`/`sys.stderr` to a new timestamped file per process under `data/logs/console/`
  (in addition to, not instead of, the terminal) — use `timestamp` in the CSVs to find the matching
  console log if you need the raw text (e.g. a stack trace) for a given episode.
- **A "fresh" run is often actually a resume.** `project.py` only builds a new `convNet` if
  `data/Network.keras` doesn't exist, and `TrainingManager.__init__` separately loads the starting
  replay buffer from one fixed-name file, `data/TrainingExamples/current_training_examples.pkl`,
  which is overwritten each episode with the full (post-trim) buffer. `data/TrainingExamples/
  Episode_N` files are a separate per-episode audit trail of just that episode's fresh self-play
  examples — written every episode, never read back, so their restart-at-0-every-run numbering
  (unlike `Episode_N.keras`) causes no ambiguity. `python project.py` is only really starting from
  scratch if both `data/Network.keras` and `current_training_examples.pkl` are absent.
- **Board bit layout is row-major but visually flipped.** `UTTTBoard`'s `x`/`o`/eligible-subboard
  bitboards index cells as `move = subboard*9 + local_cell` (both subboard and local_cell in
  standard row-major 0-8 order), but `symbol_array_representation()`'s output array is that layout
  with both axes reversed (`[::-1, ::-1]`) to match the orientation `print()` and the network's
  training data have always used. Any new code reading/writing the bitboards directly (not through
  `get_array_representation()`) needs to reason in the *unflipped* row-major indexing; anything
  consuming the `(9,9,4)` array tensor sees the *flipped* orientation.
