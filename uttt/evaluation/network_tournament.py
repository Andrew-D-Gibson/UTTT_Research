import os
import re
import csv
import glob
import shutil
import time
import multiprocessing as mp
from datetime import datetime

import numpy as np

from uttt.evaluation.elo import ELOManager
from uttt.simulation.tournament import init_tournament_worker, play_tournament_game
from uttt.config import config
from uttt.paths import NETWORKS_DIR, TOURNAMENT_LOG_PATH, TOURNAMENT_SNAPSHOT_ROOT


def discover_checkpoints(networks_dir=NETWORKS_DIR, prefix='Episode'):
    # Scans networks_dir for <prefix>_N.keras files (self-play checkpoints written by
    # uttt/training/manager.py). Prints an episode -> mtime table so a mixed-run
    # situation (mtime decreasing as episode number increases, meaning a
    # lower-numbered episode from a newer run overwrote a higher-numbered one from
    # an older run) is visible before anything else happens.
    pattern = re.compile(rf'{re.escape(prefix)}_(\d+)\.keras$')
    found = {}
    for path in glob.glob(os.path.join(networks_dir, f'{prefix}_*.keras')):
        match = pattern.search(path)
        if match is None:
            continue
        episode = int(match.group(1))
        found[episode] = (path, os.path.getmtime(path))

    print(f'Found {len(found)} {prefix}_* checkpoints in {networks_dir}/:')
    prev_mtime = None
    for episode in sorted(found):
        path, mtime = found[episode]
        stamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        flag = ''
        if prev_mtime is not None and mtime < prev_mtime:
            flag = '  <-- older than the previous episode; likely a different/restarted run'
        print(f'  {prefix}_{episode:<4} {stamp}{flag}')
        prev_mtime = mtime

    return found


def select_ladder(found, target_count=12, min_episode=None, max_episode=None):
    # Evenly subsamples the discovered episode numbers down to target_count entries,
    # always keeping the first and last. min_episode/max_episode optionally restrict
    # the pool first, e.g. to isolate one training run after eyeballing the mtime
    # table printed by discover_checkpoints().
    episodes = sorted(found)
    if min_episode is not None:
        episodes = [e for e in episodes if e >= min_episode]
    if max_episode is not None:
        episodes = [e for e in episodes if e <= max_episode]

    if not episodes:
        raise ValueError('No checkpoints left after applying min_episode/max_episode filters.')

    if len(episodes) <= target_count:
        selected = episodes
    else:
        idx = np.linspace(0, len(episodes) - 1, target_count)
        selected = sorted({episodes[int(round(i))] for i in idx})

    print(f'\nSelected {len(selected)} checkpoints for the ladder: {selected}')
    return {episode: found[episode] for episode in selected}


def snapshot_checkpoints(selected, snapshot_root=TOURNAMENT_SNAPSHOT_ROOT, prefix='Episode'):
    # Copies just the selected checkpoint files into a fresh timestamped folder, so
    # the tournament never reads a file a concurrently-running training run is
    # mid-write on, and results stay reproducible even if the originals are later
    # overwritten by episode number reuse in a new run.
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_dir = os.path.join(snapshot_root, stamp)
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshotted = {}
    for episode, (path, _mtime) in selected.items():
        dest = os.path.join(snapshot_dir, f'{prefix}_{episode}.keras')
        shutil.copy2(path, dest)
        snapshotted[episode] = dest

    print(f'Snapshotted {len(snapshotted)} checkpoints to {snapshot_dir}/')
    return snapshotted


def build_network_specs(snapshotted, prefix='Episode'):
    # Specs, not loaded agents: workers each build their own copy of every agent from
    # these specs (see simulate_tournament_games.init_tournament_worker), so the main
    # process never needs to load a single TF model itself - it only ever plays the
    # bookkeeping role of tracking names/Elo, exactly like TrainingManager never touches
    # game logic and only aggregates what its self-play/gating workers report back.
    # 'network' agents are ProbabilisticNetworkMCTSAgent (samples moves proportional to
    # visit counts), not a deterministic argmax agent: two deterministic agents playing
    # a fixed color order would replay bit-for-bit identical games every time, so
    # repeated tournament games between the same pair would add no information.
    specs = [
        {'kind': 'network', 'name': f'{prefix}_{episode}', 'path': snapshotted[episode], 'depth': None}
        for episode in sorted(snapshotted)
    ]
    print(f'Prepared {len(specs)} network agent specs: {[s["name"] for s in specs]}')
    return specs


def build_raw_mcts_specs(depths):
    # Network-free MCTS baselines (pure random rollout, no policy/value network) at
    # fixed search depths - same depths uttt/evaluation/elo.py's own baseline calibration uses. These
    # ride in the same Elo pool as the network checkpoints so the tournament answers
    # not just "which episode is best" but "is the network adding anything over plain
    # search at all," which is the more fundamental sanity check.
    specs = [{'kind': 'raw_mcts', 'name': f'RawMCTS_{depth}', 'path': None, 'depth': depth} for depth in depths]
    print(f'Prepared {len(specs)} raw MCTS baseline specs: {[s["name"] for s in specs]}')
    return specs


def _agent_kind_and_episode(name):
    if name.startswith('Episode_'):
        return 'network', name.split('_', 1)[1]
    return 'raw_mcts', ''


def run_tournament(agent_specs, iterations, num_processes, log_path=TOURNAMENT_LOG_PATH):
    # Parallelized the same way TrainingManager parallelizes self-play/gating: a
    # multiprocessing.Pool of workers (each independently built via
    # init_tournament_worker, one full agent roster per worker rather than per game)
    # plays games concurrently and reports results through a single shared
    # Manager().Queue(), while this main process is the only place Elo state lives
    # and is ever mutated - it applies each ELOManager update as results arrive,
    # same as run_gating() is the only place candidate_wins/champion_wins get tallied.
    #
    # Pairing selection (which two agents play each game) doesn't depend on Elo, so
    # every game's pairing can be decided up front and handed to the pool exactly like
    # run_self_play() submits all of its games before draining any results. Elo
    # updates, however, do depend on the *current* rating of both agents - since games
    # now complete in parallel (non-deterministic order) rather than strictly in
    # submission order, each update uses whatever ratings are current at the moment
    # its result arrives. This is the standard "rating period" approximation used to
    # parallelize Elo and converges to the same ratings as the sequential version over
    # many random pairings; it just no longer reproduces one exact sequential ordering.
    ratings = [{'name': spec['name'], 'elo': 1200.0} for spec in agent_specs]

    print('\nInitial ELOs:')
    for rating in ratings:
        print(f'{rating["name"]}: {rating["elo"]}')

    elo_history = np.empty((len(ratings), -(-iterations // 10)))  # ceil(iterations/10)

    print(f'\n--- Tournament: {iterations} games ({num_processes} processes) ---')
    start_time = time.time()

    # Force 'spawn' regardless of platform default: see the matching comment in
    # uttt/training/manager.py - the default 'fork' start method on Linux clones
    # already-initialized TensorFlow state into workers, which breaks
    # configure_cpu_worker()'s thread-count calls.
    mp_ctx = mp.get_context('spawn')
    mp_manager = mp_ctx.Manager()
    progress_queue = mp_manager.Queue()

    pool = mp_ctx.Pool(num_processes, initializer=init_tournament_worker, initargs=(agent_specs,))
    for game_index in range(iterations):
        i, j = (int(x) for x in np.random.choice(len(ratings), size=2, replace=False))
        pool.apply_async(play_tournament_game, args=(i, j, game_index, progress_queue))
    pool.close()

    games_done = 0
    while games_done < iterations:
        _pid, i, j, _game_index, result, duration = progress_queue.get()

        if games_done % 10 == 0:
            elo_history[:, games_done // 10] = [r['elo'] for r in ratings]

        exp = ELOManager.expected_result(ratings[i]['elo'], ratings[j]['elo'])
        actual = 1 if result == 1 else (0 if result == -1 else 0.5)
        delta = ELOManager.delta_elo(exp, actual)
        ratings[i]['elo'] += delta
        ratings[j]['elo'] -= delta

        games_done += 1
        outcome = 'draw' if result == 0 else (f'{ratings[i]["name"]} win' if result == 1 else f'{ratings[j]["name"]} win')
        elapsed = time.time() - start_time
        eta = (elapsed / games_done) * (iterations - games_done)
        print(f'  [{games_done}/{iterations}] {ratings[i]["name"]} vs {ratings[j]["name"]}: {outcome} '
              f'({duration:.1f}s) - elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s')

    pool.join()

    print('\nFinal ELOs:')
    for rating in ratings:
        print(f'{rating["name"]}: {rating["elo"]:.1f}')

    ranked = sorted(ratings, key=lambda r: r['elo'], reverse=True)

    print('\n--- Final ranking ---')
    for rank, rating in enumerate(ranked, start=1):
        print(f'{rank}. {rating["name"]}: {rating["elo"]:.1f}')
    print(f'\nBest overall: {ranked[0]["name"]} (Elo {ranked[0]["elo"]:.1f})')

    with open(log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'kind', 'episode', 'final_elo'])
        for rating in ranked:
            kind, episode = _agent_kind_and_episode(rating['name'])
            writer.writerow([rating['name'], kind, episode, rating['elo']])
    print(f'Wrote final standings to {log_path}')

    history_path = os.path.splitext(log_path)[0] + '_history.csv'
    with open(history_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration'] + [rating['name'] for rating in ratings])
        for col in range(elo_history.shape[1]):
            writer.writerow([col * 10] + list(elo_history[:, col]))
    print(f'Wrote Elo trajectory to {history_path}')

    return ranked


def run_default():
    TARGET_LADDER_SIZE = 12
    ITERATIONS = 192
    RAW_MCTS_DEPTHS = [16, 64, 256, 1024, 4096]   # same spread as the ELO baseline calibration

    print(
         "Note: network agents search with whatever uttt/config.py's mcts.search_depth "
         "currently is - same budget as real evaluation play, so results reflect "
         "actual playing strength rather than a cheaper stand-in. Raw MCTS baselines "
         "use their own fixed depths regardless of uttt/config.py.\n"
     )

    network_specs = []
    found = discover_checkpoints(prefix='Episode')
    if found:
        selected = select_ladder(found, target_count=TARGET_LADDER_SIZE)
        snapshotted = snapshot_checkpoints(selected, prefix='Episode')
        network_specs += build_network_specs(snapshotted, prefix='Episode')

    raw_specs = build_raw_mcts_specs(RAW_MCTS_DEPTHS)
    run_tournament(network_specs + raw_specs, iterations=ITERATIONS,
                   num_processes=config['self_play']['num_of_processes'])
