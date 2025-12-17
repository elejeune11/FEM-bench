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
    if not isinstance(n_nodes, (int, np.integer)):
        raise TypeError('n_nodes must be an integer')
    if n_nodes < 0:
        raise ValueError('n_nodes must be >= 0')
    total_dofs = 6 * n_nodes
    all_dofs = np.arange(total_dofs, dtype=int)
    fixed_list = []
    if boundary_conditions is None:
        boundary_conditions = {}
    try:
        items = boundary_conditions.items()
    except AttributeError:
        raise TypeError('boundary_conditions must be a dict-like mapping')
    for node, mask in items:
        if not isinstance(node, (int, np.integer)):
            raise TypeError('Node indices must be integers')
        if not 0 <= int(node) < n_nodes:
            raise ValueError(f'Node index {node} out of valid range [0, {n_nodes - 1}]')
        mask_arr = np.asarray(mask, dtype=bool).ravel()
        if mask_arr.size != 6:
            raise ValueError(f'Boundary condition for node {node} must have 6 boolean entries')
        base = 6 * int(node) + np.arange(6, dtype=int)
        fixed_list.extend(base[mask_arr].tolist())
    if fixed_list:
        fixed = np.unique(np.array(fixed_list, dtype=int))
    else:
        fixed = np.array([], dtype=int)
    if total_dofs == 0:
        free = np.array([], dtype=int)
    elif fixed.size == 0:
        free = all_dofs
    else:
        mask_free = np.ones(total_dofs, dtype=bool)
        mask_free[fixed] = False
        free = all_dofs[mask_free]
    return (fixed, free)