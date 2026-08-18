import numpy as np

# The 8 elements of the dihedral group D4 (identity, 3 rotations, 4 reflections),
# indexed 0-7. Applying symmetry k to a (9,9,...) board array is valid because the
# board is a self-similar 3x3-of-3x3 grid: a rotation/reflection of the whole 9x9
# grid decomposes into the *same* operation applied simultaneously to the subboard
# layout and to each subboard's local cell layout (both are 3x3, so the coordinate
# math for "which subboard" and "which cell within it" is identical).
NUM_SYMMETRIES = 8

# Precompute the fixed bijection between move index (subboard*9 + local_cell, both
# row-major - see uttt_board.py's board bit layout) and the (row, col) coordinates
# of that move in the *visual* (9,9) array space produced by
# UTTTBoard.symbol_array_representation(). This mirrors that method's
# reshape(3,3,3,3).transpose(0,2,1,3).reshape(9,9) followed by [::-1, ::-1].
_MOVE_ROW = np.empty(81, dtype=int)
_MOVE_COL = np.empty(81, dtype=int)
for _move in range(81):
    _subboard, _local = divmod(_move, 9)
    _sub_row, _sub_col = divmod(_subboard, 3)
    _local_row, _local_col = divmod(_local, 3)
    _MOVE_ROW[_move] = 8 - (3 * _sub_row + _local_row)
    _MOVE_COL[_move] = 8 - (3 * _sub_col + _local_col)

_MOVE_INDEX_GRID = np.empty((9, 9), dtype=int)
_MOVE_INDEX_GRID[_MOVE_ROW, _MOVE_COL] = np.arange(81)

# move (subboard*9 + local_cell, row-major) -> flat index into a (81,) vector built
# by flattening a (9,9) array-space grid in row-major order (array.reshape(81)).
# Used to build the fixed permutation matrix that reindexes a policy head's
# array-space logits into move order - see architectures.py's hierarchicalResNet().
MOVE_TO_ARRAY_FLAT_INDEX = _MOVE_ROW * 9 + _MOVE_COL


def apply_board_symmetry(array, symmetry_index):
    """Apply dihedral symmetry `symmetry_index` (0-7) to a (9,9,...) board array,
    transforming only the leading two (spatial) axes."""
    if not (0 <= symmetry_index < NUM_SYMMETRIES):
        raise ValueError(f'symmetry_index must be in [0, {NUM_SYMMETRIES}), got {symmetry_index}')

    transformed = array
    if symmetry_index >= 4:
        transformed = np.swapaxes(transformed, 0, 1)
    return np.ascontiguousarray(np.rot90(transformed, k=symmetry_index % 4, axes=(0, 1)))


def apply_move_vector_symmetry(move_vector, symmetry_index):
    """Apply the same dihedral symmetry to an 81-length vector indexed by move
    (subboard*9 + local_cell), e.g. MCTS search probabilities."""
    grid = move_vector[_MOVE_INDEX_GRID]
    grid = apply_board_symmetry(grid, symmetry_index)
    return grid[_MOVE_ROW, _MOVE_COL]
