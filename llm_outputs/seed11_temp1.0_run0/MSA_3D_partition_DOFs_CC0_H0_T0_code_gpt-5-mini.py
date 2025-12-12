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
    if not isinstance(n_nodes, int) or n_nodes < 0:
        raise ValueError('n_nodes must be a non-negative integer')
    total_dofs = 6 * n_nodes
    if boundary_conditions is None:
        boundary_conditions = {}
    if not isinstance(boundary_conditions, dict):
        raise ValueError('boundary_conditions must be a dict mapping node->6-length iterable of bools')
    fixed_indices = []
    for (node, bc) in boundary_conditions.items():
        if not isinstance(node, int):
            raise ValueError('node indices must be integers')
        if node < 0 or node >= n_nodes:
            raise ValueError(f'node index {node} out of valid range [0, {n_nodes - 1}]')
        try:
            length = len(bc)
        except TypeError:
            raise ValueError(f'boundary condition for node {node} must be an iterable of length 6')
        if length != 6:
            raise ValueError(f'boundary condition for node {node} must have length 6')
        for local_dof in range(6):
            try:
                is_fixed = bool(np.asarray(bc)[local_dof])
            except Exception:
                try:
                    is_fixed = bool(bc[local_dof])
                except Exception:
                    raise ValueError(f'invalid boundary condition entry at node {node}, index {local_dof}')
            if is_fixed:
                fixed_indices.append(6 * node + local_dof)
    if len(fixed_indices) == 0:
        fixed_arr = np.array([], dtype=int)
    else:
        fixed_arr = np.unique(np.array(fixed_indices, dtype=int))
    if total_dofs == 0:
        free_arr = np.array([], dtype=int)
    else:
        all_dofs = np.arange(total_dofs, dtype=int)
        if fixed_arr.size == 0:
            free_arr = all_dofs
        else:
            free_arr = np.setdiff1d(all_dofs, fixed_arr, assume_unique=True)
    return (fixed_arr, free_arr)