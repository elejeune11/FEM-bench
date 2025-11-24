def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    Assumes global Cartesian coordinates and right-hand rule for orientation.
    the degrees of freedom (DOFs) into free and fixed sets, and solves the system
    for nodal displacements and support reactions.
    The system is solved using a partitioned approach. Displacements are computed
    at free DOFs, and true reaction forces (including contributions from both
    stiffness and applied loads) are computed at fixed DOFs. The system is only
    solved if the free-free stiffness matrix is well-conditioned
    (i.e., condition number < 1e16). If the matrix is ill-conditioned or singular,
    a ValueError is raised.
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : (3,) array or None
                Optional unit vector to define the local z-direction for transformation matrix orientation.
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.
    Returns
    -------
    u : (6 * N,) ndarray
        Global displacement vector. Entries are ordered as [UX, UY, UZ, RX, RY, RZ] for each node.
        Displacements are computed only at free DOFs; fixed DOFs are set to zero.
    r : (6 * N,) ndarray
        Global reaction force and moment vector. Nonzero values are present only at fixed DOFs
        and reflect the net support reactions, computed as internal elastic forces minus applied loads.
    Raises
    ------
    ValueError
        If the free-free stiffness matrix is ill-conditioned and the system cannot be reliably solved.
    Helper Functions Used
    ---------------------
        Assembles the global 6N x 6N stiffness matrix using local beam element stiffness and transformations.
        Assembles the global load vector from nodal force/moment data.
        Identifies fixed and free degrees of freedom based on boundary condition flags.
        Solves the reduced system for displacements and computes reaction forces.
        Raises a ValueError if the system is ill-conditioned.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must have shape (N, 3).')
    n_nodes = node_coords.shape[0]
    if n_nodes == 0:
        return (np.zeros(0, dtype=float), np.zeros(0, dtype=float))
    for ele in elements:
        if 'node_i' not in ele or 'node_j' not in ele:
            raise ValueError("Each element must contain 'node_i' and 'node_j'.")
        ni = int(ele['node_i'])
        nj = int(ele['node_j'])
        if not 0 <= ni < n_nodes or not 0 <= nj < n_nodes:
            raise IndexError('Element node indices out of range.')
    bc_bool: dict[int, np.ndarray] = {}
    for (n, flags) in boundary_conditions.items():
        if not 0 <= int(n) < n_nodes:
            raise IndexError('Boundary condition node index out of range.')
        arr = np.asarray(flags, dtype=int).astype(bool)
        if arr.shape != (6,):
            raise ValueError('Boundary condition flags must be length 6 per node.')
        bc_bool[int(n)] = arr
    nodal_loads_clean: dict[int, np.ndarray] = {}
    for (n, load) in nodal_loads.items():
        if not 0 <= int(n) < n_nodes:
            raise IndexError('Nodal load node index out of range.')
        arr = np.asarray(load, dtype=float)
        if arr.shape != (6,):
            raise ValueError('Nodal load must be length 6 per node.')
        nodal_loads_clean[int(n)] = arr
    K_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(nodal_loads_clean, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    if free.size == 0:
        u = np.zeros(6 * n_nodes, dtype=float)
        r = np.zeros_like(u)
        r[fixed] = -P_global[fixed]
        return (u, r)
    (u, r) = linear_solve(P_global, K_global, fixed, free)
    return (u, r)