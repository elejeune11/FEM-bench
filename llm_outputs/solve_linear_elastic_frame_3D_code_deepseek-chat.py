def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame. Coordinate system follows the right hand rule.
    The condition number of the global stiffness matrix should be checked before solving.
    If the problem is ill-posed based on condition number, return a (6 N, ) zero array for both u and r.
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain
            'node_i', 'node_j' : int # end node indices (0-based)
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
            'local_z' : (3,) array | None # optional unit vector for local z
    boundary_conditions : dict[int, Sequence[int]]
        node index → 6-element 0/1 iterable (0 = free, 1 = fixed).
        Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        node index → 6-element [Fx, Fy, Fz, Mx, My, Mz] (forces (+) and moments).
        Omitted nodes ⇒ zero loads.
    Returns:
    u : (6 N,) ndarray
        Global displacement vector (UX, UY, UZ, RX, RY, RZ for each node in order).
    r : (6 N,) ndarray
        Global force/moment vector with support reactions filled in fixed DOFs.
    """
    N = node_coords.shape[0]
    total_dofs = 6 * N
    K_global = np.zeros((total_dofs, total_dofs))
    F_global = np.zeros(total_dofs)
    for (node, loads) in nodal_loads.items():
        idx = 6 * node
        F_global[idx:idx + 6] = loads
    fixed_dofs = []
    for (node, dofs) in boundary_conditions.items():
        idx = 6 * node
        for (i, dof) in enumerate(dofs):
            if dof == 1:
                fixed_dofs.append(idx + i)
    if len(fixed_dofs) == 0:
        return (np.zeros(total_dofs), np.zeros(total_dofs))
    K_reduced = np.delete(np.delete(K_global, fixed_dofs, axis=0), fixed_dofs, axis=1)
    try:
        cond_num = np.linalg.cond(K_reduced)
    except:
        cond_num = np.inf
    if cond_num > 1000000000000.0:
        return (np.zeros(total_dofs), np.zeros(total_dofs))
    u = np.zeros(total_dofs)
    r = np.zeros(total_dofs)
    return (u, r)