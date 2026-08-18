import numpy as np
import tensorflow as tf

from uttt.board.symmetry import MOVE_TO_ARRAY_FLAT_INDEX

def convNet():
    ultimate_tic_tac_toe_input = tf.keras.layers.Input(shape=(9,9,4), name='uttt_input')

    conv_1 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', name='conv_1')(ultimate_tic_tac_toe_input)
    batchnorm = tf.keras.layers.BatchNormalization()(conv_1)
    conv_2 = tf.keras.layers.Conv2D(128, (3,3), strides=(3, 3), activation='relu', name='conv_2')(batchnorm)
    batchnorm = tf.keras.layers.BatchNormalization()(conv_2)

    flatten = tf.keras.layers.Flatten()(batchnorm)

    dense_1 = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(flatten)
    dense_2 = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(dense_1)

    # Raw logits, not softmax: masking illegal moves + softmax happens outside the
    # model, at MCTS/inference time (see uttt/search/mcts.py's check_network), which
    # indexes this output as legal_logits and softmaxes only over legal moves. A
    # softmax activation here would double-softmax (once over all 81 moves here,
    # again over the legal subset there), which isn't a valid probability
    # distribution over legal moves.
    policy_output = tf.keras.layers.Dense(81, name='policy_output')(dense_2)
    value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(dense_2)

    model = tf.keras.models.Model(inputs=ultimate_tic_tac_toe_input, outputs=[policy_output, value_output], name='convNet')

    # Compile model
    losses = {
        'policy_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        'value_output': 'mse'
    }

    # 'accuracy' only makes sense for the softmax policy head; the tanh-activated
    # scalar value head would be scored as an (essentially meaningless) exact-match
    # check on a continuous value, so it's only requested for policy_output here.
    model.compile(optimizer='Adam', loss=losses, metrics={'policy_output': tf.keras.metrics.CategoricalAccuracy(name='accuracy')})
    return model


# --- hierarchicalResNet() -----------------------------------------------------
#
# Named constants mirror the style of the reference architecture this design was
# based on. Kept modest in scale (not AlphaZero-sized) per the design brief.
NUM_CHANNELS = 4
META_FILTERS = 128
NUM_RES_BLOCKS = 3
L2_WEIGHT = 1e-4

POLICY_META_BRANCH_FILTERS = 32     # channels after reducing meta_features for broadcast
POLICY_LOCAL_3X3_FILTERS = 32       # shared per-subboard 3x3 conv branch
POLICY_LOCAL_1X1_FILTERS = 32       # per-cell 1x1 conv branch (own 4 input values only)
POLICY_COMBINE_FILTERS = 64         # 1x1 conv bottleneck before the final per-cell logit

VALUE_HEAD_FILTERS = 32
VALUE_HEAD_DENSE_UNITS = 256


def _l2():
    return tf.keras.regularizers.l2(L2_WEIGHT)


def _conv_bn_relu(x, filters, kernel_size, name, strides=1, padding='same'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                                use_bias=False, kernel_regularizer=_l2(), name=f'{name}_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(x)
    x = tf.keras.layers.ReLU(name=f'{name}_relu')(x)
    return x


def _residual_block(x, filters, name):
    # Standard AlphaZero-style residual block: two 3x3 convs (stride 1, same padding
    # so shape is preserved) with BN, skip-add before the second ReLU. Operates
    # entirely on the (3,3,filters) meta-board representation - stride-1 3x3 convs
    # here mix information *between* subboards (that's the point, this is the
    # meta-board reasoning stage), unlike the stride-3 stage-1 conv below.
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                kernel_regularizer=_l2(), name=f'{name}_conv1')(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(y)
    y = tf.keras.layers.ReLU(name=f'{name}_relu1')(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                kernel_regularizer=_l2(), name=f'{name}_conv2')(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(y)
    y = tf.keras.layers.Add(name=f'{name}_add')([shortcut, y])
    y = tf.keras.layers.ReLU(name=f'{name}_relu2')(y)
    return y


def _build_policy_reindex_matrix():
    # array-space flat index (row*9+col, row-major flatten of the (9,9) grid) -> the
    # policy head's per-cell logit conv produces one value per array-space cell,
    # flattened row-major. This frozen Dense layer reindexes that (81,) vector into
    # game-move order (subboard*9+local_cell) via a fixed permutation matrix, so
    # every other module (MCTS, TrainingExample, agents) can keep indexing
    # policy_output the same way they already index board.find_moves()/search_probs.
    #
    # A Dense computes y = x @ kernel, i.e. y[move] = sum_i x[i] * kernel[i, move].
    # We want y[move] = x[MOVE_TO_ARRAY_FLAT_INDEX[move]], so kernel must be 1 at
    # (MOVE_TO_ARRAY_FLAT_INDEX[move], move) and 0 elsewhere.
    #
    # Implemented as a frozen Dense (not a Lambda/tf.gather) because Lambda layers
    # with closures fail to deserialize by default in this environment
    # ("arbitrary code execution... disallowed by default"), and every network
    # load site in this codebase (self_play.py, gating.py, tournament.py,
    # manager.py's clone_network, agent.py, interface.py) calls
    # tf.keras.models.load_model() without custom_objects/safe_mode=False. Dense is
    # a built-in, universally-serializable layer type, so this sidesteps that
    # problem entirely.
    matrix = np.zeros((81, 81), dtype=np.float32)
    matrix[MOVE_TO_ARRAY_FLAT_INDEX, np.arange(81)] = 1.0
    return matrix


def hierarchicalResNet():
    inputs = tf.keras.layers.Input(shape=(9, 9, NUM_CHANNELS), name='uttt_input')

    # --- Stage 1: per-subboard aggregation, no cross-subboard mixing ----------
    # stride=(3,3) + padding='valid' means each output cell is a function of
    # exactly one 3x3 subboard block and nothing else - the only place in the
    # network where subboard boundaries are respected by construction rather than
    # by training. (9,9,C) -> (3,3,META_FILTERS).
    x = tf.keras.layers.Conv2D(META_FILTERS, 3, strides=(3, 3), padding='valid',
                                use_bias=False, kernel_regularizer=_l2(), name='stage1_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stage1_bn')(x)
    x = tf.keras.layers.ReLU(name='stage1_relu')(x)

    # --- Meta-board reasoning: residual tower on the (3,3) grid ----------------
    for i in range(NUM_RES_BLOCKS):
        x = _residual_block(x, META_FILTERS, name=f'meta_res{i}')
    meta_features = x  # (3, 3, META_FILTERS)

    # --- Policy head: three branches feeding a per-cell logit ------------------
    #
    # Branch 1 - meta broadcast: every cell in a subboard sees that subboard's
    # meta-board-aware feature vector (full-board strategic context).
    meta_reduced = _conv_bn_relu(meta_features, POLICY_META_BRANCH_FILTERS, 1, name='policy_meta_reduce')
    # UpSampling2D(nearest) repeats meta_reduced[i,j] across the 3x3 array-space
    # block starting at (3i,3j) - exactly the same subboard block stage1_conv
    # collapsed it from, so the broadcast lines up with array-space geometry.
    meta_broadcast = tf.keras.layers.UpSampling2D(size=(3, 3), name='policy_meta_broadcast')(meta_reduced)

    # Branch 2 - shared local 3x3: one Conv2D/BN/ReLU triple, reused (weight-shared)
    # across all 9 subboards, so it learns one "read this subboard's local pattern"
    # function applied identically everywhere. Needs explicit per-subboard slicing
    # because a plain stride-3 Conv2D would collapse each subboard to a single
    # value per filter, giving every empty cell in a subboard identical logits
    # (verified this is an architectural ceiling, not a training-difficulty issue,
    # before ruling it out).
    local3x3_conv = tf.keras.layers.Conv2D(POLICY_LOCAL_3X3_FILTERS, 3, padding='same', use_bias=False,
                                            kernel_regularizer=_l2(), name='policy_local3x3_conv')
    local3x3_bn = tf.keras.layers.BatchNormalization(name='policy_local3x3_bn')
    local3x3_relu = tf.keras.layers.ReLU(name='policy_local3x3_relu')

    row_bands = []
    for sub_row in range(3):
        patches = []
        for sub_col in range(3):
            r0, c0 = sub_row * 3, sub_col * 3
            patch = inputs[:, r0:r0 + 3, c0:c0 + 3, :]
            patch = local3x3_conv(patch)
            patch = local3x3_bn(patch)
            patch = local3x3_relu(patch)
            patches.append(patch)
        row_bands.append(tf.keras.layers.Concatenate(axis=2, name=f'policy_local3x3_row{sub_row}')(patches))
    local_3x3 = tf.keras.layers.Concatenate(axis=1, name='policy_local3x3_full')(row_bands)  # (9,9,filters)

    # Branch 3 - per-cell 1x1: sees only that cell's own 4 input channels, no
    # neighboring cells at all (a 1x1 conv is inherently per-position).
    local_1x1 = _conv_bn_relu(inputs, POLICY_LOCAL_1X1_FILTERS, 1, name='policy_local1x1')

    # Combine: stay in (9,9,C) feature-map space and let 1x1 convs (weight-shared,
    # position-independent) produce the final per-cell logit, rather than flattening
    # into a giant Dense(81) - keeps per-cell resolution without an unnecessarily
    # large, position-dependent parameter block.
    policy_combined = tf.keras.layers.Concatenate(axis=-1, name='policy_branch_concat')(
        [meta_broadcast, local_3x3, local_1x1])
    policy_combined = _conv_bn_relu(policy_combined, POLICY_COMBINE_FILTERS, 1, name='policy_combine')
    policy_logits_map = tf.keras.layers.Conv2D(1, 1, padding='same', kernel_regularizer=_l2(),
                                                name='policy_logit_conv')(policy_combined)  # (9,9,1)

    # Flatten row-major (matches MOVE_TO_ARRAY_FLAT_INDEX's array-space convention),
    # then reindex array-space -> game-move order via the frozen permutation Dense.
    policy_array_space = tf.keras.layers.Reshape((81,), name='policy_flatten')(policy_logits_map)
    policy_reindex = tf.keras.layers.Dense(
        81, use_bias=False, trainable=False, name='policy_reindex',
        kernel_initializer=tf.keras.initializers.Constant(_build_policy_reindex_matrix()),
    )(policy_array_space)

    # Raw logits, not softmax: masking illegal moves + softmax happens outside the
    # model, at MCTS/inference time (see uttt/search/mcts.py), since only the
    # caller knows which moves are legal for a given board state.

    # --- Value head --------------------------------------------------------------
    value = _conv_bn_relu(meta_features, VALUE_HEAD_FILTERS, 1, name='value_head')
    value = tf.keras.layers.Flatten(name='value_flatten')(value)
    value = tf.keras.layers.Dense(VALUE_HEAD_DENSE_UNITS, activation='relu',
                                   kernel_regularizer=_l2(), name='value_dense')(value)
    value_output = tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=_l2(),
                                          name='value_dense_out')(value)

    model = tf.keras.models.Model(
        inputs=inputs,
        outputs={'policy_output': policy_reindex, 'value_output': value_output},
        name='hierarchicalResNet',
    )

    losses = {
        'policy_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        'value_output': tf.keras.losses.MeanSquaredError(),
    }
    metrics = {
        'policy_output': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        'value_output': tf.keras.metrics.MeanAbsoluteError(name='value_mae'),
    }
    model.compile(optimizer='Adam', loss=losses, metrics=metrics)
    return model


# Registry used by build_network() to hot-swap architectures via config.py's
# config['network']['architecture'] instead of editing launcher code.
ARCHITECTURES = {
    'convNet': convNet,
    'hierarchicalResNet': hierarchicalResNet,
}


def build_network(name=None):
    # name defaults to config['network']['architecture'] so every fresh-build call
    # site (currently just project.py) picks up the config switch without each one
    # importing uttt.config itself.
    if name is None:
        from uttt.config import config
        name = config['network']['architecture']
    try:
        return ARCHITECTURES[name]()
    except KeyError:
        raise ValueError(f"Unknown network architecture '{name}'. Valid options: {list(ARCHITECTURES)}")
