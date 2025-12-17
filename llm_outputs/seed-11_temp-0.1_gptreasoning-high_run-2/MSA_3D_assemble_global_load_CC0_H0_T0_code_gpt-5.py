def MSA_3D_assemble_global_load_CC0_H0_T0(nodal_loads: dict, n_nodes: int):
    """
    Assemble the global nodal load vector for a 3D linear-elastic frame structure.
    Constructs the global right-hand-side vector (P) for the equilibrium equation:
        K * u = P
    where K is the assembled global stiffness matrix and u is the global displacement
    vector. Each node contributes up to six DOFs corresponding to forces and moments
    in the global Cartesian frame.
    Parameters
    ----------
    nodal_loads : dict[int, array-like of float]
        Mapping from node index (0-based) to a 6-component load vector:
            [F_x, F_y, F_z, M_x, M_y, M_z]
        representing forces (N) and moments (N·m) applied at that node in
        **global coordinates**. Nodes not listed are assumed to have zero loads.
    n_nodes : int
        Total number of nodes in the structure. Must be consistent with the
        indexing used in `nodal_loads`.
    Returns
    -------
    P : (6 * n_nodes,) ndarray of float
        Global load vector containing all nodal forces and moments.
        DOF ordering per node: [UX, UY, UZ, RX, RY, RZ].
        Entries for unconstrained or unloaded nodes are zero.
    Notes
    -----
        DOFs for node n → [6*n, 6*n+1, 6*n+2, 6*n+3, 6*n+4, 6*n+5].
    """
    import numpy as np
    try:
        n = int(n_nodes)
    except Exception as e:
        raise TypeError('n_nodes must be an integer') from e
    if n < 0:
        raise ValueError('n_nodes must be non-negative')
    P = np.zeros(6 * n, dtype=float)
    if nodal_loads is None:
        return P
    try:
        items_iter = nodal_loads.items()
    except AttributeError as e:
        raise TypeError('nodal_loads must be a mapping of node_index -> 6-component load vector') from e
    for node_idx, load_vec in items_iter:
        if isinstance(node_idx, (int, np.integer)):
            idx = int(node_idx)
        elif isinstance(node_idx, float) and node_idx.is_integer():
            idx = int(node_idx)
        else:
            raise TypeError(f"Node index '{node_idx}' must be an integer (0-based).")
        if idx < 0 or idx >= n:
            raise IndexError(f'Node index {idx} out of range for n_nodes={n}.')
        arr = np.asarray(load_vec, dtype=float).reshape(-1)
        if arr.size != 6:
            raise ValueError(f'Nodal load vector for node {idx} must have exactly 6 components; got {arr.size}.')
        start = 6 * idx
        P[start:start + 6] = arr
    return P