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
    total_dofs = 6 * n_nodes
    if total_dofs == 0:
        return (np.array([], dtype=int), np.array([], dtype=int))
    fixed_mask_global = np.zeros(total_dofs, dtype=bool)
    if boundary_conditions is None:
        boundary_conditions = {}
    try:
        items = boundary_conditions.items()
    except AttributeError:
        raise TypeError('boundary_conditions must be a dict-like mapping')
    for (node, mask) in items:
        if not isinstance(node, int):
            raise TypeError('Node indices must be integers')
        if node < 0 or node >= n_nodes:
            raise ValueError(f'Node index {node} out of valid range [0, {n_nodes - 1}]')
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != (6,):
            if arr.ndim != 1 or arr.size != 6:
                raise ValueError(f'Boundary condition for node {node} must be a 6-length iterable')
            arr = arr.reshape(6)
        base = 6 * node
        for j in range(6):
            if arr[j]:
                fixed_mask_global[base + j] = True
    fixed = np.flatnonzero(fixed_mask_global).astype(int, copy=False)
    free = np.flatnonzero(~fixed_mask_global).astype(int, copy=False)
    return (fixed, free)