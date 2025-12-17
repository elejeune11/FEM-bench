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
    if not isinstance(n_nodes, int):
        raise TypeError('n_nodes must be an integer')
    if n_nodes < 0:
        raise ValueError('n_nodes must be >= 0')
    if boundary_conditions is None:
        bc = {}
    elif isinstance(boundary_conditions, dict):
        bc = boundary_conditions
    else:
        raise TypeError('boundary_conditions must be a dict mapping node index to 6-length iterable of bools')
    total_dofs = 6 * n_nodes
    is_fixed = np.zeros(total_dofs, dtype=bool)
    for node, flags_raw in bc.items():
        if not isinstance(node, int):
            raise TypeError('Node indices must be integers')
        if node < 0 or node >= n_nodes:
            raise ValueError(f'Node index {node} out of range for n_nodes={n_nodes}')
        flags = np.asarray(flags_raw, dtype=bool).ravel()
        if flags.size != 6:
            raise ValueError(f'Boundary condition for node {node} must have length 6')
        base = 6 * node
        is_fixed[base:base + 6] |= flags
    fixed = np.flatnonzero(is_fixed).astype(int, copy=False)
    free = np.flatnonzero(~is_fixed).astype(int, copy=False)
    return (fixed, free)