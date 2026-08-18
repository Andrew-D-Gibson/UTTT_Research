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
    def __init__(self, request_queue, response_timeout=60.0):
        self._request_queue = request_queue
        # duplex=False: this client only ever reads, the inference server only ever
        # writes, on a given request/response leg.
        self._client_conn, self._server_conn = mp.Pipe(duplex=False)
        self._response_timeout = response_timeout

    def __call__(self, board_arrays, training=False):
        # check_network always calls with a batch of exactly one board - unwrap it
        # here so callers on both ends of the queue only ever deal in single boards.
        board = board_arrays[0]
        self._request_queue.put((board, self._server_conn))

        if not self._client_conn.poll(self._response_timeout):
            raise TimeoutError(
                f'No response from inference server within {self._response_timeout}s '
                '- it may have died, or the request queue is badly backed up.'
            )

        policy, value = self._client_conn.recv()
        return {'policy_output': policy, 'value_output': value}


def run_inference_server(network_path, request_queue, gpu_id=None,
                          max_batch_size=64, max_wait_s=0.005):
    # Must happen before TensorFlow is imported/initialized in this process.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if gpu_id is None else str(gpu_id)

    import tensorflow as tf

    network = tf.keras.models.load_model(network_path, compile=False)

    log_prefix = f'[inference server pid={os.getpid()} gpu={gpu_id}]'
    total_requests = 0
    total_batches = 0

    while True:
        item = request_queue.get()
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
            conn.send((policies[i:i + 1], values[i:i + 1]))

        total_requests += len(batch_boards)
        total_batches += 1

        if stop_after_batch:
            break

    avg_batch = total_requests / total_batches if total_batches else 0
    print(f'{log_prefix} done - {total_requests} requests in {total_batches} batches '
          f'(avg batch size {avg_batch:.1f})', flush=True)
