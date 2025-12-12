def partition_degrees_of_freedom(boundary_conditions: dict, n_nodes: int):
    """
    Partitions the degrees of freedom (DOFs) into fixed and free sets for a 3D frame structure.
    Parameters
    ----------
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping node indices to 6-element boolean arrays, where `True` indicates the DOF is fixed.
        Nodes not listed are assumed fully free.
    n_nodes : int
        Total number of nodes in the structure.
    Returns
    -------
    fixed : ndarray of int
        Sorted array of fixed DOF indices.
    free : ndarray of int
        Sorted array of free DOF indices.
    """
    n_dof = n_nodes * 6
    fixed = []
    for n in range(n_nodes):
        flags = boundary_conditions.get(n)
        if flags is not None:
            fixed.extend([6 * n + i for (i, f) in enumerate(flags) if f])
    fixed = np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)
    return (fixed, free)