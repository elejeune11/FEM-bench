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
    if n_nodes is None:
        raise TypeError('n_nodes must be provided as an integer')
    try:
        n_nodes = int(n_nodes)
    except Exception:
        raise TypeError('n_nodes must be an integer')
    if n_nodes < 0:
        raise ValueError('n_nodes must be non-negative')
    total_dofs = 6 * n_nodes
    fixed_flags = np.zeros(total_dofs, dtype=bool)
    if boundary_conditions is None:
        boundary_conditions = {}
    if not hasattr(boundary_conditions, 'items'):
        raise TypeError('boundary_conditions must be a mapping of node->iterable of 6 booleans')
    for (node, bc) in boundary_conditions.items():
        if not isinstance(node, (int, np.integer)):
            raise TypeError('boundary_conditions keys must be integer node indices')
        node_idx = int(node)
        if node_idx < 0 or node_idx >= n_nodes:
            raise IndexError(f'Node index {node_idx} out of range [0, {max(n_nodes - 1, 0)}]')
        arr = np.asarray(bc).ravel()
        if arr.size != 6:
            raise ValueError(f'Boundary condition for node {node_idx} must have length 6')
        mask = np.asarray(arr, dtype=bool)
        if mask.any():
            base = 6 * node_idx
            true_locs = np.flatnonzero(mask)
            fixed_flags[base + true_locs] = True
    fixed = np.flatnonzero(fixed_flags).astype(int)
    free = np.flatnonzero(~fixed_flags).astype(int)
    return (fixed, free)