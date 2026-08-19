# Library imports
import os
import csv
import json
import time
import queue
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import pickle
from datetime import datetime

# Function imports
from uttt.board.symmetry import apply_board_symmetry, apply_move_vector_symmetry, NUM_SYMMETRIES
from uttt.simulation.self_play import simulate_self_play_games, init_self_play_worker
from uttt.simulation.gating import simulate_gating_games, init_gating_worker
from uttt.inference.server import run_inference_server
from uttt.worker import enable_gpu_memory_growth

from uttt.config import config
from uttt.paths import (
    NETWORK_PATH,
    NETWORKS_DIR,
    TRAINING_EXAMPLES_DIR,
    CURRENT_TRAINING_EXAMPLES_PATH,
    TRAINING_LOG_PATH,
    GATING_LOG_PATH,
    EPOCH_HISTORY_DIR,
    ensure_data_dirs,
)
from uttt.run_logging import snapshot_config, new_session_id


class WorkerPoolStalledError(RuntimeError):
    """Raised by TrainingManager._drain_progress_queue when a self-play/gating
    worker pool goes silent for longer than config['self_play']['stall_timeout_s']."""


class TrainingManager:
    def __init__(self, session_id=None):
        ensure_data_dirs()
        snapshot_config()

        self.session_id = session_id or new_session_id()

        # This process trains the candidate (model.fit()) and shares its GPU(s) with the
        # inference-server processes (uttt/inference/server.py), which also use growth
        # mode - without this, TF's default "grab ~90%+ of the first GPU up front"
        # behavior here competes with them for the same memory instead of coexisting.
        # A no-op if project.py's own earlier call already covered it (idempotent as
        # long as no GPU has been used yet) - kept here too since TrainingManager can be
        # constructed directly without going through project.py.
        enable_gpu_memory_growth()

        self.network = tf.keras.models.load_model(NETWORK_PATH)

        self.training_examples = self.load_latest_training_examples()
        self.log_path = TRAINING_LOG_PATH
        self.log_fields = [
            'episode', 'timestamp', 'session_id', 'config_version', 'champion_episode',
            'self_play_games', 'self_play_avg_plies', 'self_play_duration_s',
            'new_examples', 'buffer_size',
            'training_sample_size', 'training_duration_s',
            'epochs_run', 'epochs_configured', 'stopped_early',
            'train_loss', 'val_loss',
            'train_policy_loss', 'val_policy_loss',
            'train_value_loss', 'val_value_loss',
            'train_policy_acc', 'val_policy_acc',
            'episode_duration_s',
        ]

        # Force 'spawn' regardless of platform default: on Linux, mp.Pool()'s default
        # 'fork' start method clones the parent's already-initialized TensorFlow state
        # (the champion network is loaded before any Pool is created), and TF raises
        # "Intra op parallelism cannot be modified after initialization" if a forked
        # child then tries to set thread counts (self-play/gating workers no longer do
        # this - see uttt/worker.py's seed_worker_rng() - but the inference server
        # processes still do, via os.environ['CUDA_VISIBLE_DEVICES'] in
        # run_inference_server, uttt/inference/server.py, which must also happen
        # before TF initializes in that process). 'spawn' starts each worker/server as
        # a clean interpreter so TF is uninitialized until that code runs. macOS
        # already defaults to 'spawn', which is why this only surfaces on Linux.
        self.mp_ctx = mp.get_context('spawn')

        # Long-lived so every episode can hand out a fresh Queue from it without
        # spinning up a new manager subprocess each time.
        self.mp_manager = self.mp_ctx.Manager()

        self.episodes_run = 0
        self.promotions = 0
        self.last_promoted_episode = None

    def load_latest_training_examples(self, path=CURRENT_TRAINING_EXAMPLES_PATH):
        # Jump-starts the replay buffer from the master buffer file instead of always
        # starting empty. Episode_N files are a separate per-episode audit trail of
        # just that episode's fresh examples (see train()) - they're never read back
        # here, only this one fixed-name file is.
        if not os.path.exists(path):
            print('No existing training examples found - starting with an empty replay buffer.')
            return []

        with open(path, 'rb') as file:
            examples = pickle.load(file)

        print(f'Resuming replay buffer from {path} ({len(examples)} training examples)')

        return examples

    def log_metrics(self, row):
        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.log_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_gating_result(self, episode, timestamp, champion_episode, duration,
                           candidate_wins, draws, champion_wins, win_rate, promoted):
        path = GATING_LOG_PATH
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['episode', 'timestamp', 'session_id', 'config_version', 'champion_episode',
                                  'gating_duration_s', 'candidate_wins', 'draws', 'champion_wins',
                                  'win_rate', 'promoted'])
            writer.writerow([episode, timestamp, self.session_id, config['version'], champion_episode,
                              duration, candidate_wins, draws, champion_wins, win_rate, promoted])

    def clone_network(self, network):
        # fit() mutates weights in place, so training the champion directly would
        # destroy it before we know whether the result is actually better. Clone via
        # the same save/load round-trip used everywhere else in this codebase (rather
        # than clone_model/get_weights) so the compiled optimizer state comes along
        # for free.
        tmp_path = os.path.join(NETWORKS_DIR, '_candidate_tmp.keras')
        network.save(tmp_path)
        clone = tf.keras.models.load_model(tmp_path)
        os.remove(tmp_path)
        return clone

    def train_on_examples(self, network, examples):
        board_train = np.empty((len(examples),9,9,4))
        search_probs_train = np.empty((len(examples),81))
        eval_train = np.empty((len(examples),1))

        # Each example is stored in the buffer in one canonical orientation (see
        # TrainingExample); reorienting here rather than at self-play record time means
        # the same stored example gets a fresh random symmetry every time it's sampled
        # into a training batch, so its exposure varies episode to episode instead of
        # being permanently pinned to whatever orientation it was recorded under.
        augment = config['training']['random_symmetry_augmentation']
        for i, example in enumerate(examples):
            board_array = example.board_array
            search_probs = example.search_probs

            if augment:
                symmetry_index = np.random.randint(NUM_SYMMETRIES)
                board_array = apply_board_symmetry(board_array, symmetry_index)
                search_probs = apply_move_vector_symmetry(search_probs, symmetry_index)

            board_train[i,:,:,:] = board_array
            search_probs_train[i,:] = search_probs
            eval_train[i] = example.reward

        # A single shuffled pass over a large sample of the buffer, with a real
        # validation split, instead of hammering 16 disjoint 512-example slices
        # for up to 50 epochs each (which just overfits to whichever slice is
        # currently in view and wastes wall-clock time better spent on self-play).
        loss_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['training_patience'],
            restore_best_weights=True,
        )

        history = network.fit(
            board_train,
            [search_probs_train, eval_train],
            batch_size=config['training']['minibatch_size'],
            epochs=config['training']['training_epochs'],
            validation_split=0.1,
            shuffle=True,
            callbacks=[loss_callback],
            verbose=2)  # one line/epoch, no progress bar - keeps a long fit() from looking hung

        h = history.history
        train_loss, val_loss = h['loss'][-1], h['val_loss'][-1]
        train_policy_loss, val_policy_loss = h['policy_output_loss'][-1], h['val_policy_output_loss'][-1]
        train_value_loss, val_value_loss = h['value_output_loss'][-1], h['val_value_output_loss'][-1]
        train_policy_acc, val_policy_acc = h['policy_output_accuracy'][-1], h['val_policy_output_accuracy'][-1]

        epochs_run = len(h['loss'])
        epochs_configured = config['training']['training_epochs']
        stopped_early = epochs_run < epochs_configured
        epoch_note = f'stopped early at {epochs_run}/{epochs_configured}' if stopped_early else f'ran all {epochs_run}'

        # Comparing train vs val on each component separately (rather than just the combined
        # loss) is what actually shows overfitting: the two heads can diverge in opposite
        # directions and that's invisible in a single blended number.
        print(f' Loss   train/val: {train_loss:.4f} / {val_loss:.4f}')
        print(f' Policy train/val: acc {train_policy_acc:.3f}/{val_policy_acc:.3f}, loss {train_policy_loss:.4f}/{val_policy_loss:.4f}')
        print(f' Value  train/val loss: {train_value_loss:.4f} / {val_value_loss:.4f}')
        print(f' Epochs: {epoch_note}')

        # Returned rather than logged here - train() assembles one full per-episode row
        # (self-play stats, timing, these metrics) and writes it in a single log_metrics call.
        # 'epoch_history' carries the full per-epoch curve (h itself only has the last epoch's
        # values pulled out above) for train() to dump separately - it isn't a CSV column.
        return {
            'epochs_run': epochs_run,
            'epochs_configured': epochs_configured,
            'stopped_early': stopped_early,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_policy_loss': train_policy_loss, 'val_policy_loss': val_policy_loss,
            'train_value_loss': train_value_loss, 'val_value_loss': val_value_loss,
            'train_policy_acc': train_policy_acc, 'val_policy_acc': val_policy_acc,
            'epoch_history': {key: [float(v) for v in values] for key, values in h.items()},
        }


    def print_settings(self):
        games_per_episode = config['self_play']['num_of_processes'] * config['self_play']['num_of_self_play_games_per_process']
        print('=== Training run settings ===')
        print(f"  mcts:      search_depth={config['mcts']['search_depth']}, "
              f"exploration={config['mcts']['exploration_parameter']}, "
              f"dirichlet(alpha={config['mcts']['dirichlet_alpha']}, eps={config['mcts']['dirichlet_epsilon']})")
        print(f"  self_play: {config['self_play']['num_of_processes']} processes x "
              f"{config['self_play']['num_of_self_play_games_per_process']} games/process "
              f"({games_per_episode} games/episode), "
              f"temperature_moves={config['self_play']['temperature_moves']}, "
              f"gating/testing_games={config['self_play']['num_of_testing_games']}")
        print(f"  training:  sample_size={config['training']['training_sample_size']}, "
              f"minibatch={config['training']['minibatch_size']}, "
              f"epochs<={config['training']['training_epochs']} (patience={config['training']['training_patience']}), "
              f"buffer_cap={config['training']['max_training_examples']}")
        print(f"  episodes:  {config['training']['num_of_episodes']}")

    def start_inference_server(self, network_path, gpu_id):
        # One request queue + one dedicated process owning one loaded copy of
        # `network_path`, pinned to `gpu_id` (uttt/inference/server.py). Plain
        # mp_ctx.Queue(), not self.mp_manager.Queue() - this queue is on the hot path
        # (hit once per MCTS leaf, ~512 times per move per worker), and routing that
        # through the SyncManager's proxy process (as progress_queue does, fine for
        # its much lower message rate) would add a lot of avoidable overhead. Plain
        # Queue also reliably carries the Connection objects InferenceClient passes
        # through it, which a Manager-proxied queue isn't guaranteed to preserve.
        request_queue = self.mp_ctx.Queue()
        server = self.mp_ctx.Process(
            target=run_inference_server,
            args=(network_path, request_queue, gpu_id,
                  config['inference']['max_batch_size'],
                  config['inference']['max_wait_ms'] / 1000.0),
        )
        server.start()
        return request_queue, server

    def start_inference_servers(self, network_path):
        # One queue *per distinct GPU*, not one shared queue for every server -
        # servers assigned to the same GPU (config['inference']['gpu_ids'] repeats an
        # id once per server process wanted on that GPU, e.g. [0, 0, 0, 1, 1, 1] for 3
        # servers/GPU) share that GPU's queue and naturally work-steal among
        # themselves, but each GPU needs its own queue so InferenceClient can
        # explicitly round-robin *across* GPUs. A single queue shared by every server
        # regardless of GPU has no fairness guarantee across concurrent consumer
        # processes - in practice this let one GPU's servers win the race to drain it
        # almost every time, leaving other GPUs' servers (and the GPUs themselves)
        # comparatively idle instead of splitting load evenly.
        # Returns a list of (queue, [server, ...]) groups, one per distinct GPU - pass
        # [queue for queue, _ in groups] to init_self_play_worker, and the whole list
        # to stop_inference_servers.
        configured_gpu_ids = config['inference']['gpu_ids']
        distinct_gpu_ids = list(dict.fromkeys(configured_gpu_ids)) or [None]
        server_groups = []
        for gpu_id in distinct_gpu_ids:
            servers_on_this_gpu = configured_gpu_ids.count(gpu_id) if configured_gpu_ids else 1
            queue = self.mp_ctx.Queue()
            servers = []
            for _ in range(servers_on_this_gpu):
                server = self.mp_ctx.Process(
                    target=run_inference_server,
                    args=(network_path, queue, gpu_id,
                          config['inference']['max_batch_size'],
                          config['inference']['max_wait_ms'] / 1000.0),
                )
                server.start()
                servers.append(server)
            server_groups.append((queue, servers))
        return server_groups

    def stop_inference_servers(self, server_groups):
        # One None sentinel per server in each group's queue - each server exits after
        # consuming exactly one (see run_inference_server's `if item is None: break`).
        for queue, servers in server_groups:
            for _ in servers:
                queue.put(None)
        for _, servers in server_groups:
            for server in servers:
                server.join()

    def _drain_progress_queue(self, progress_queue, total_units, async_results, on_message):
        # Bounded-wait drain: an unconditional blocking queue.get() would hang the run
        # forever, with no error and no log output, if a worker died before delivering its
        # full quota of progress messages. The stall_timeout_s check on top of that catches
        # the specific case an empty-queue/all-ready check alone can't: a worker whose OS
        # process died mid-task, since multiprocessing.Pool spawns a replacement worker for
        # future tasks but never resolves the AsyncResult that was in flight on the dead one
        # - nothing else distinguishes "still working" from "gone forever" in that case.
        stall_timeout = config['self_play']['stall_timeout_s']
        units_done = 0
        last_progress_time = time.time()

        while units_done < total_units:
            try:
                msg = progress_queue.get(timeout=5)
            except queue.Empty:
                if all(r.ready() for r in async_results):
                    break
                if time.time() - last_progress_time > stall_timeout:
                    raise WorkerPoolStalledError(
                        f'No progress from the worker pool in over {stall_timeout}s '
                        f'({units_done}/{total_units} done) - a worker likely died silently.'
                    )
                continue

            last_progress_time = time.time()
            units_done += 1
            on_message(msg, units_done)

        return units_done

    def _force_teardown(self, pool, server_groups):
        # Used only after a stall - pool.join() and the sentinel-based
        # stop_inference_servers() both assume every worker/server is still alive and
        # responsive enough to voluntarily exit, which is exactly what a stall means we
        # can no longer assume. terminate() sends SIGTERM directly instead of waiting.
        pool.terminate()
        pool.join()
        for _, servers in server_groups:
            for server in servers:
                server.terminate()
            for server in servers:
                server.join()

    def run_self_play(self):
        num_processes = config['self_play']['num_of_processes']
        games_per_process = config['self_play']['num_of_self_play_games_per_process']
        total_games = num_processes * games_per_process
        max_retries = config['self_play']['max_stall_retries']

        print(f'--- Self-play: {total_games} games ({num_processes} processes x {games_per_process}/process) ---')

        attempt = 0
        while True:
            attempt += 1
            self_play_start_time = time.time()

            progress_queue = self.mp_manager.Queue()
            server_groups = self.start_inference_servers(NETWORK_PATH)
            request_queues = [queue for queue, _ in server_groups]

            # request_queues must go through Pool's initializer/initargs, not apply_async's
            # per-task args - a raw multiprocessing.Queue can only be inherited at worker
            # process creation, not pickled through Pool's task-dispatch queue at call time
            # (see uttt/simulation/self_play.py's init_self_play_worker).
            pool = self.mp_ctx.Pool(num_processes, initializer=init_self_play_worker,
                                     initargs=(request_queues,))
            async_results = [
                pool.apply_async(simulate_self_play_games, args=(progress_queue,))
                for _ in range(num_processes)
            ]
            pool.close()

            total_plies = 0

            def on_message(msg, games_done_so_far):
                nonlocal total_plies
                pid, game_idx, worker_num_games, plies, duration = msg
                total_plies += plies
                elapsed = time.time() - self_play_start_time
                eta = (elapsed / games_done_so_far) * (total_games - games_done_so_far)
                print(f'  [{games_done_so_far}/{total_games}] worker {pid}: game {game_idx}/{worker_num_games} '
                      f'({plies} plies, {duration:.1f}s) - elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s')

            try:
                self._drain_progress_queue(progress_queue, total_games, async_results, on_message)
            except WorkerPoolStalledError as e:
                print(f'{e} Tearing down this attempt\'s pool and inference servers '
                      f'and retrying (attempt {attempt}/{max_retries + 1}).')
                self._force_teardown(pool, server_groups)
                if attempt > max_retries:
                    raise
                continue

            pool.join()
            self.stop_inference_servers(server_groups)

            # .get() (not the old callback=) so a worker exception is re-raised here instead of
            # silently under-filling the replay buffer this episode was supposed to receive.
            new_examples = []
            for r in async_results:
                new_examples.extend(r.get())
            self.training_examples.extend(new_examples)

            self_play_elapsed = time.time() - self_play_start_time
            print(f'Self-play done in {self_play_elapsed:.1f}s '
                  f'({self_play_elapsed/total_games:.1f}s/game avg, {total_plies/total_games:.1f} plies/game avg)')
            print(f'Replay buffer: {len(self.training_examples)} examples')

            return self_play_elapsed, total_games, total_plies, new_examples

    def run_gating(self, candidate):
        num_processes = config['self_play']['num_of_processes']
        total_games = config['self_play']['num_of_testing_games']
        max_retries = config['self_play']['max_stall_retries']

        # Balanced per-worker chunk sizes, skipping any that would land on zero
        # games (e.g. total_games < num_processes at today's defaults) so we don't
        # pay process-start + two-model-load cost for workers that do nothing.
        base, remainder = divmod(total_games, num_processes)
        chunk_sizes = [size for size in
                       (base + (1 if i < remainder else 0) for i in range(num_processes))
                       if size > 0]

        # Cumulative offsets so each chunk's local i%2 alternation lines up with
        # what a single sequential agent_match call over all total_games would
        # have produced - restarting parity at 0 per chunk would systematically
        # favor whichever agent goes first (see PlayerAgent.agent_match).
        start_indices = []
        cumulative = 0
        for size in chunk_sizes:
            start_indices.append(cumulative)
            cumulative += size

        # The candidate only exists in memory at this point (clone_network() +
        # train_on_examples() never touch disk) - gating workers need it on disk
        # to load independently, exactly like self-play workers already load the
        # champion from Network.keras.
        candidate_path = os.path.join(NETWORKS_DIR, '_gating_candidate.keras')
        candidate.save(candidate_path)

        try:
            print(f'--- Gating: {total_games} games ({len(chunk_sizes)} processes) - '
                  f'candidate vs champion ---')

            # Candidate and champion are different weights, so they need two separate
            # inference servers rather than sharing one - split across the configured
            # GPUs (candidate on the first, champion on the second) so gating actually
            # keeps both GPUs busy; both fall back to the same id/CPU if only one is
            # configured. Dedup (preserving order) rather than indexing gpu_ids directly -
            # self_play.gpu_ids repeats each id once per server process on that GPU (e.g.
            # [0, 0, 0, 1, 1, 1] for 3 servers/GPU), so gpu_ids[1] would still be 0.
            distinct_gpu_ids = list(dict.fromkeys(config['inference']['gpu_ids'])) or [None]
            candidate_gpu = distinct_gpu_ids[0]
            champion_gpu = distinct_gpu_ids[1] if len(distinct_gpu_ids) > 1 else distinct_gpu_ids[0]

            attempt = 0
            while True:
                attempt += 1
                gating_start_time = time.time()
                progress_queue = self.mp_manager.Queue()

                candidate_queue, candidate_server = self.start_inference_server(candidate_path, candidate_gpu)
                champion_queue, champion_server = self.start_inference_server(NETWORK_PATH, champion_gpu)
                server_groups = [(candidate_queue, [candidate_server]),
                                  (champion_queue, [champion_server])]

                pool = self.mp_ctx.Pool(len(chunk_sizes), initializer=init_gating_worker,
                                         initargs=(candidate_queue, champion_queue))
                async_results = [
                    pool.apply_async(simulate_gating_games, args=(size, start_index, progress_queue))
                    for size, start_index in zip(chunk_sizes, start_indices)
                ]
                pool.close()

                def on_message(msg, games_done_so_far):
                    pid, game_idx, worker_num_games, outcome, duration = msg
                    elapsed = time.time() - gating_start_time
                    eta = (elapsed / games_done_so_far) * (total_games - games_done_so_far)
                    print(f'  [{games_done_so_far}/{total_games}] worker {pid}: game {game_idx}/{worker_num_games} '
                          f'({outcome}, {duration:.1f}s) - elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s')

                try:
                    self._drain_progress_queue(progress_queue, total_games, async_results, on_message)
                except WorkerPoolStalledError as e:
                    print(f'{e} Tearing down this attempt\'s pool and inference servers '
                          f'and retrying (attempt {attempt}/{max_retries + 1}).')
                    self._force_teardown(pool, server_groups)
                    if attempt > max_retries:
                        raise
                    continue

                pool.join()
                self.stop_inference_servers(server_groups)

                # .get() (not a callback) so a worker exception is re-raised here
                # instead of silently under-counting the tally that decides promotion.
                totals = [r.get() for r in async_results]
                candidate_wins = sum(t[0] for t in totals)
                draws = sum(t[1] for t in totals)
                champion_wins = sum(t[2] for t in totals)

                gating_elapsed = time.time() - gating_start_time
                print(f'Gating done in {gating_elapsed:.1f}s ({gating_elapsed/total_games:.1f}s/game avg)')

                return candidate_wins, draws, champion_wins, gating_elapsed
        finally:
            if os.path.exists(candidate_path):
                os.remove(candidate_path)

    def train(self):
        self.print_settings()

        for episode in range(config['training']['num_of_episodes']):
            episode_start_time = time.time()
            episode_timestamp = datetime.now().isoformat(timespec='seconds')

            print(f'\n\n===== Episode {episode} =====')

            # The champion self-play/gating actually played against this episode - captured
            # before gating below can update last_promoted_episode, so it reflects what was
            # really in data/Network.keras during this episode, not what it becomes after.
            champion_episode = self.last_promoted_episode if self.last_promoted_episode is not None else -1

            self_play_duration, self_play_games, self_play_total_plies, new_examples = self.run_self_play()

            # If we have too many examples, remove from the front
            if len(self.training_examples) > config['training']['max_training_examples']:
                self.training_examples = self.training_examples[len(self.training_examples) - config['training']['max_training_examples']:]

            # Save training examples now, right after self-play finishes and before
            # training the candidate - so a crash/interruption during training or
            # gating doesn't lose an episode's worth of fresh self-play games.
            # Episode_N is a permanent per-episode audit trail of just this episode's
            # fresh examples (never trimmed, never read back in); the master buffer
            # file is what actually gets resumed from and reflects the trimmed cap.
            with open(os.path.join(TRAINING_EXAMPLES_DIR, f'Episode_{episode}'), 'wb') as file:
                pickle.dump(new_examples, file)

            with open(CURRENT_TRAINING_EXAMPLES_PATH, 'wb') as file:
                pickle.dump(self.training_examples, file)

            # Train a candidate cloned from the current champion on a large shuffled
            # sample of the whole buffer. The champion itself is never mutated here,
            # so a rejected candidate can just be discarded below.
            print('--- Training candidate ---')
            training_start_time = time.time()

            sample_size = np.min([config['training']['training_sample_size'], len(self.training_examples)])
            training_sample = np.random.choice(self.training_examples, sample_size, replace=False)
            print(f'Training on {sample_size} examples (sampled from {len(self.training_examples)} in buffer)')

            candidate = self.clone_network(self.network)
            metrics = self.train_on_examples(candidate, training_sample)

            # Full per-epoch curve for this episode's fit() call - the CSV row below only
            # keeps the final epoch's numbers, this keeps the whole trajectory for later
            # inspection (e.g. within-episode overfitting) without bloating training_log.csv.
            epoch_history = metrics.pop('epoch_history')
            with open(os.path.join(EPOCH_HISTORY_DIR, f'Episode_{episode}.json'), 'w') as file:
                json.dump(epoch_history, file)

            training_duration = time.time() - training_start_time
            print(f'Training took {training_duration:.1f}s')

            # Gate: the candidate must beat the champion head-to-head to be promoted.
            # A tie or a loss keeps the existing champion, and no checkpoint is saved
            # for this episode - Networks/ ends up sparse, reflecting only verified
            # improvements rather than every training round.
            candidate_wins, draws, champion_wins, gating_duration = self.run_gating(candidate)

            # Require a clear margin over decisive (non-drawn) games rather than a bare
            # majority - with num_of_testing_games in the tens, wins-vs-losses margins of
            # 1-2 games are well within noise and shouldn't flip the champion.
            decisive_games = candidate_wins + champion_wins
            win_rate = candidate_wins / decisive_games if decisive_games > 0 else 0.0
            promoted = win_rate >= config['self_play']['promotion_win_rate']
            verdict = 'PROMOTED' if promoted else 'REJECTED (keeping champion)'
            print(f'Candidate vs champion W/D/L: {candidate_wins}/{draws}/{champion_wins} '
                  f'(win rate {win_rate:.1%} of decisive games) -> {verdict}')

            self.log_gating_result(episode, episode_timestamp, champion_episode, gating_duration,
                                    candidate_wins, draws, champion_wins, win_rate, promoted)

            self.episodes_run += 1
            if promoted:
                self.network = candidate
                self.network.save(os.path.join(NETWORKS_DIR, f'Episode_{episode}.keras'))
                self.network.save(NETWORK_PATH)
                self.promotions += 1
                self.last_promoted_episode = episode

            last_promo_note = f'episode {self.last_promoted_episode}' if self.last_promoted_episode is not None else 'none yet'
            print(f'Promotions so far: {self.promotions}/{self.episodes_run} episodes (last: {last_promo_note})')

            episode_duration = time.time() - episode_start_time
            print(f'===== Episode {episode} done in {episode_duration:.1f}s =====')

            self.log_metrics({
                'episode': episode,
                'timestamp': episode_timestamp,
                'session_id': self.session_id,
                'config_version': config['version'],
                'champion_episode': champion_episode,
                'self_play_games': self_play_games,
                'self_play_avg_plies': self_play_total_plies / self_play_games if self_play_games else 0,
                'self_play_duration_s': self_play_duration,
                'new_examples': len(new_examples),
                'buffer_size': len(self.training_examples),
                'training_sample_size': sample_size,
                'training_duration_s': training_duration,
                'episode_duration_s': episode_duration,
                **metrics,
            })
