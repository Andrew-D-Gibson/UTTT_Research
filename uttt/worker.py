import os
import random
import numpy as np


def configure_cpu_worker():
    # Turn off the GPU for multiprocessing, so tensorflow doesn't over-allocate GPU memory
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Local import, not top-of-file: this module is also imported by self-play/gating
    # *worker* processes (for seed_worker_rng()) and by uttt/inference/server.py, none of
    # which should pay TensorFlow's import cost (meaningfully slow, plus real per-process
    # memory) just because they happen to share this file with a function that needs it.
    import tensorflow as tf

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


def enable_gpu_memory_growth():
    # Must run before ANY op touches a GPU in this process - building or loading a model,
    # calling fit(), etc. all count. TF raises "Physical devices cannot be modified after
    # being initialized" if this is called after a device's first use, so every entry
    # point that might build/load a model needs to call this first, before doing so
    # (project.py does, before build_network(); TrainingManager.__init__ and
    # run_inference_server also call it defensively in case a model gets built/loaded
    # before either of those runs). Without this, TF's default behavior is to
    # pre-allocate ~90%+ of a GPU's memory to the first process that touches it, which
    # starves every other process sharing that GPU once more than one does (multiple
    # inference-server processes per GPU, and/or this training process itself).
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
