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
    if not isinstance(n_nodes, (int, np.integer)) or isinstance(n_nodes, bool):
        raise TypeError('n_nodes must be an integer')
    if n_nodes < 0:
        raise ValueError('n_nodes must be >= 0')
    total_dofs = 6 * n_nodes
    if boundary_conditions is None:
        bc = {}
    elif isinstance(boundary_conditions, dict):
        bc = boundary_conditions
    else:
        raise TypeError('boundary_conditions must be a dict mapping node index to a 6-length iterable of booleans')
    fixed_indices = []
    for (node, mask) in bc.items():
        if not isinstance(node, (int, np.integer)) or isinstance(node, bool):
            raise TypeError('Node indices must be integers')
        if node < 0 or node >= n_nodes:
            raise ValueError(f'Node index {node} out of range for n_nodes={n_nodes}')
        arr = np.asarray(mask, dtype=bool).ravel()
        if arr.size != 6:
            raise ValueError(f'Boundary condition for node {node} must have length 6')
        base = 6 * int(node)
        for j in range(6):
            if bool(arr[j]):
                fixed_indices.append(base + j)
    if fixed_indices:
        fixed = np.unique(np.asarray(fixed_indices, dtype=int))
    else:
        fixed = np.asarray([], dtype=int)
    if total_dofs == 0:
        if len(bc) != 0:
            raise ValueError('boundary_conditions specifies nodes but n_nodes=0')
        free = np.asarray([], dtype=int)
        return (fixed, free)
    all_dofs = np.arange(total_dofs, dtype=int)
    if fixed.size == 0:
        free = all_dofs
    elif fixed.size == total_dofs:
        free = np.asarray([], dtype=int)
    else:
        free = np.setdiff1d(all_dofs, fixed, assume_unique=True)
    return (fixed, free)