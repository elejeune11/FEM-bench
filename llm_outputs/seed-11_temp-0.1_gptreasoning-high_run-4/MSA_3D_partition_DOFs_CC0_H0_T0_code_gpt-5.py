def MSA_3D_partition_DOFs_CC0_H0_T0(boundary_conditions: dict, n_nodes: int):
    """
    Partition global degrees of freedom (DOFs) into fixed and free sets for a 3D frame.
    Purpose
    -------
    For a 3D frame with 6 DOFs per node (3 translations and 3 rotations), this routine
    builds the global index sets of fixed and free DOFs used to assemble, partition,
    and solve the linear system. The global DOF ordering per node is:
        [UX, UY, UZ, RX, RY, RZ].
    Global indices are 0-based and assigned as: DOF_index = 6*node + local_dof.
    Parameters
    ----------
    boundary_conditions : dict[int, array-like of bool]
        Maps node index (0-based, in [0, n_nodes-1]) to a 6-length iterable of booleans
        indicating which DOFs are fixed at that node (True = fixed, False = free).
        Nodes not present in the dict are assumed to have all DOFs free.
    n_nodes : int
        Total number of nodes in the structure (N ≥ 0).
    Returns
    -------
    fixed : ndarray of int, shape (n_fixed,)
        Sorted, unique global indices of fixed DOFs (0-based). May be empty if no DOFs are fixed.
    free : ndarray of int, shape (6*N - n_fixed,)
        Sorted, unique global indices of free DOFs (0-based). Disjoint from `fixed`.
        The union of `fixed` and `free` covers all DOFs: {0, 1, ..., 6*N - 1}.
    """
    import numpy as np
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)):
        raise TypeError('n_nodes must be a non-negative integer.')
    if n_nodes < 0:
        raise ValueError('n_nodes must be non-negative.')
    total_dofs = 6 * n_nodes
    if boundary_conditions is None or len(boundary_conditions) == 0:
        fixed = np.empty(0, dtype=int)
        free = np.arange(total_dofs, dtype=int) if total_dofs > 0 else np.empty(0, dtype=int)
        return (fixed, free)
    if not isinstance(boundary_conditions, dict):
        raise TypeError('boundary_conditions must be a dict mapping node index to 6-length iterable of bools.')
    fixed_chunks = []
    for node, flags in boundary_conditions.items():
        if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
            raise TypeError('Node indices must be integers (0-based).')
        node_idx = int(node)
        if node_idx < 0 or node_idx >= n_nodes:
            raise ValueError(f'Node index {node_idx} out of range [0, {n_nodes - 1}].')
        try:
            arr = np.asarray(flags, dtype=bool).reshape(-1)
        except Exception as e:
            raise TypeError(f'Invalid boundary condition flags for node {node_idx}: {e}')
        if arr.size != 6:
            raise ValueError(f'Boundary condition flags for node {node_idx} must have length 6.')
        local_fixed = np.nonzero(arr)[0]
        if local_fixed.size:
            base = 6 * node_idx
            fixed_chunks.append(base + local_fixed.astype(int))
    if fixed_chunks:
        fixed = np.unique(np.concatenate(fixed_chunks).astype(int))
    else:
        fixed = np.empty(0, dtype=int)
    if total_dofs == 0:
        free = np.empty(0, dtype=int)
    elif fixed.size == 0:
        free = np.arange(total_dofs, dtype=int)
    elif fixed.size == total_dofs:
        free = np.empty(0, dtype=int)
    else:
        all_dofs = np.arange(total_dofs, dtype=int)
        free = np.setdiff1d(all_dofs, fixed, assume_unique=False)
    return (fixed, free)