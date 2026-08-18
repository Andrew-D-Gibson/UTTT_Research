import os
import random
import numpy as np
import tensorflow as tf


def configure_cpu_worker():
    # Turn off the GPU for multiprocessing, so tensorflow doesn't over-allocate GPU memory
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Each worker process gets its own default TF thread pool, which oversubscribes
    # the CPU badly once several worker processes run at once (num_of_processes in
    # uttt/config.py). Capping each worker to a single thread keeps total thread count
    # sane and avoids the resulting slowdown. Must be set before any TF op runs.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Give each worker its own RNG state so parallel games don't end up correlated
    # (matters if multiprocessing's start method is ever 'fork' instead of 'spawn').
    random.seed(os.getpid())
    np.random.seed(os.getpid())


def seed_worker_rng():
    # Same RNG-seeding half as configure_cpu_worker(), for workers that talk to a
    # remote InferenceClient (uttt/inference/server.py) instead of loading their own
    # network - those workers never import TensorFlow, so there's no GPU to hide and
    # no per-process TF thread pool to cap.
    random.seed(os.getpid())
    np.random.seed(os.getpid())
