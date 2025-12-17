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
    if not isinstance(n_nodes, (int, np.integer)):
        raise TypeError('n_nodes must be an integer')
    if n_nodes < 0:
        raise ValueError('n_nodes must be a non-negative integer')
    P = np.zeros(6 * n_nodes, dtype=float)
    if nodal_loads is None:
        return P
    items = nodal_loads.items() if isinstance(nodal_loads, dict) else getattr(nodal_loads, 'items', lambda: None)()
    if items is None:
        raise TypeError('nodal_loads must be a dict-like mapping from node index to load vector')
    for node_idx, load_vec in items:
        if not isinstance(node_idx, (int, np.integer)):
            raise TypeError(f'Node index {node_idx!r} must be an integer')
        if node_idx < 0 or node_idx >= n_nodes:
            raise IndexError(f'Node index {node_idx} out of bounds for n_nodes={n_nodes}')
        arr = np.asarray(load_vec, dtype=float).reshape(-1)
        if arr.size != 6:
            raise ValueError(f'Load vector for node {node_idx} must have 6 components, got {arr.size}')
        start = 6 * node_idx
        P[start:start + 6] = arr
    return P