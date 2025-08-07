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
    dof_per_node = 6
    total_dof = N * dof_per_node
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        A = element['A']
        I_y = element['I_y']
        I_z = element['I_z']
        J = element['J']
        local_z = element.get('local_z', None)
        K_element = np.zeros((12, 12))
        dof_indices = np.array([node_i * dof_per_node, node_i * dof_per_node + 1, node_i * dof_per_node + 2, node_i * dof_per_node + 3, node_i * dof_per_node + 4, node_i * dof_per_node + 5, node_j * dof_per_node, node_j * dof_per_node + 1, node_j * dof_per_node + 2, node_j * dof_per_node + 3, node_j * dof_per_node + 4, node_j * dof_per_node + 5])
        for i in range(12):
            for j in range(12):
                K_global[dof_indices[i], dof_indices[j]] += K_element[i, j]
    for (node, loads) in nodal_loads.items():
        for i in range(dof_per_node):
            F_global[node * dof_per_node + i] += loads[i]
    fixed_dofs = []
    for (node, bc) in boundary_conditions.items():
        for i in range(dof_per_node):
            if bc[i] == 1:
                fixed_dofs.append(node * dof_per_node + i)
    if np.linalg.cond(K_global) > 10000000000.0:
        return (np.zeros(total_dof), np.zeros(total_dof))
    free_dofs = np.setdiff1d(np.arange(total_dof), fixed_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F_global[free_dofs]
    u = np.zeros(total_dof)
    u[free_dofs] = np.linalg.solve(K_ff, F_f)
    r = np.dot(K_global, u) - F_global
    return (u, r)