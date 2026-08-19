import os
import time
import queue
import multiprocessing as mp

import numpy as np


class InferenceClient:
    # Drop-in stand-in for a loaded Keras network. Implements the same
    # __call__(board_arrays, training=False) interface MCTS.check_network already
    # calls (uttt/search/mcts.py), so MCTS can't tell this apart from a real model -
    # no changes are needed to MCTS's tree-walk/recursion to use this instead.
    def __init__(self, request_queues, response_timeout=60.0):
        # One queue per distinct GPU (servers sharing a GPU share that GPU's queue -
        # see TrainingManager.start_inference_servers). Explicitly round-robin across
        # them here rather than relying on multiprocessing.Queue to fairly arbitrate
        # between multiple concurrent consumer processes when several GPUs' servers
        # all drain one shared queue - it doesn't: whichever server process happens to
        # win the wakeup race keeps winning it, which in practice left one GPU doing
        # most of the work and another sitting nearly idle instead of splitting evenly.
        self._request_queues = request_queues
        self._next_queue_index = 0
        # duplex=False: this client only ever reads, the inference server only ever
        # writes, on a given request/response leg.
        self._client_conn, self._server_conn = mp.Pipe(duplex=False)
        self._response_timeout = response_timeout

    def __call__(self, board_arrays, training=False):
        # check_network always calls with a batch of exactly one board - unwrap it
        # here so callers on both ends of the queue only ever deal in single boards.
        board = board_arrays[0]
        request_queue = self._request_queues[self._next_queue_index]
        self._next_queue_index = (self._next_queue_index + 1) % len(self._request_queues)
        request_queue.put((board, self._server_conn))

        if not self._client_conn.poll(self._response_timeout):
            raise TimeoutError(
                f'No response from inference server within {self._response_timeout}s '
                '- it may have died, or the request queue is badly backed up.'
            )

        policy, value = self._client_conn.recv()
        return {'policy_output': policy, 'value_output': value}


# Raised when the process on the other end of a Pipe/Queue has already died -
# e.g. a self-play worker killed mid-request. Left uncontained, any one of these
# used to crash the entire inference server process, which took every *other*
# worker sharing this GPU's queue down with it too (they'd eventually hit
# InferenceClient's 60s response timeout and die themselves). Caught narrowly
# below (not a bare except) so a real bug elsewhere - a bad board, a network()
# failure - still fails loud instead of being silently swallowed.
_DEAD_PEER_ERRORS = (BrokenPipeError, EOFError, ConnectionResetError,
                      ConnectionRefusedError, FileNotFoundError)


def run_inference_server(network_path, request_queue, gpu_id=None,
                          max_batch_size=64, max_wait_s=0.005):
    # Must happen before TensorFlow is imported/initialized in this process - including
    # transitively, which is why uttt.worker (imports tensorflow at module level) is
    # imported here, locally, rather than at this module's top level: uttt.inference.server
    # is also imported by self-play/gating *worker* processes (for InferenceClient), which
    # must never import TensorFlow at all, and a top-level import here would do so before
    # this line even runs.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if gpu_id is None else str(gpu_id)

    from uttt.worker import enable_gpu_memory_growth
    import tensorflow as tf

    # Must also happen before any op touches the GPU (before loading the model below).
    # Without this, the first process to touch a physical GPU pre-allocates ~90%+ of its
    # memory for itself by default - fine with one process per GPU, but fatal once
    # multiple inference-server processes share a GPU (gpu_id repeated in
    # config['inference']['gpu_ids']): the first one to initialize starves the rest, which
    # then fail (often as a confusing "RESOURCE_EXHAUSTED... cuDNN engine profiling
    # failure" rather than an obvious allocation error).
    enable_gpu_memory_growth()

    network = tf.keras.models.load_model(network_path, compile=False)

    log_prefix = f'[inference server pid={os.getpid()} gpu={gpu_id}]'
    total_requests = 0
    total_batches = 0
    dropped_requests = 0

    while True:
        try:
            item = request_queue.get()
        except _DEAD_PEER_ERRORS as e:
            # The client that queued this request is already gone (its fd handoff
            # never completed) - the malformed item is already off the queue by
            # the time unpickling raised, so just move on to the next one.
            print(f'{log_prefix} dropped an orphaned request ({e!r}) - continuing', flush=True)
            dropped_requests += 1
            continue

        if item is None:
            break

        batch_boards = [item[0]]
        batch_conns = [item[1]]
        deadline = time.monotonic() + max_wait_s
        stop_after_batch = False

        # Opportunistically drain more requests that are already (or about to be)
        # waiting, up to max_batch_size boards or max_wait_s since this batch
        # started - whichever comes first, so a lone late request never blocks a
        # batch that's otherwise ready to go.
        while len(batch_boards) < max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining)
            except queue.Empty:
                break
            except _DEAD_PEER_ERRORS as e:
                print(f'{log_prefix} dropped an orphaned request ({e!r}) - continuing', flush=True)
                dropped_requests += 1
                continue

            if item is None:
                stop_after_batch = True
                break

            batch_boards.append(item[0])
            batch_conns.append(item[1])

        board_arrays = np.stack(batch_boards)
        outputs = network(board_arrays, training=False)
        policies = np.asarray(outputs['policy_output'])
        values = np.asarray(outputs['value_output'])

        for i, conn in enumerate(batch_conns):
            try:
                conn.send((policies[i:i + 1], values[i:i + 1]))
            except _DEAD_PEER_ERRORS as e:
                # That one requester died between submitting and us responding -
                # drop its result and keep serving the rest of the batch instead
                # of taking the whole server down with it.
                print(f'{log_prefix} could not deliver a result to a dead client ({e!r}) - dropping it', flush=True)
                dropped_requests += 1

        total_requests += len(batch_boards)
        total_batches += 1

        if stop_after_batch:
            break

    avg_batch = total_requests / total_batches if total_batches else 0
    print(f'{log_prefix} done - {total_requests} requests in {total_batches} batches '
          f'(avg batch size {avg_batch:.1f}, {dropped_requests} dropped)', flush=True)
