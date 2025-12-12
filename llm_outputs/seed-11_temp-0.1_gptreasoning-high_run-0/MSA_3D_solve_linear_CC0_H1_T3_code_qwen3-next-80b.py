def MSA_3D_solve_linear_CC0_H1_T3(P_global: np.ndarray, K_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    n_dof = 6 * n_nodes
    fixed_dofs = np.zeros(n_dof, dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        start_dof = node_idx * 6
        fixed_dofs[start_dof:start_dof + 6] = bc
    free_dofs = ~fixed_dofs
    n_free = np.sum(free_dofs)
    n_fixed = np.sum(fixed_dofs)
    if n_free == 0:
        u = np.zeros(n_dof)
        r = P_global.copy()
        return (u, r)
    K_ff = K_global[free_dofs][:, free_dofs]
    K_fs = K_global[free_dofs][:, fixed_dofs]
    K_sf = K_global[fixed_dofs][:, free_dofs]
    K_ss = K_global[fixed_dofs][:, fixed_dofs]
    P_free = P_global[free_dofs]
    P_fixed = P_global[fixed_dofs]
    if n_free > 0:
        cond_kff = np.linalg.cond(K_ff)
        if cond_kff >= 1e+16:
            raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned')
    u_free = np.linalg.solve(K_ff, P_free)
    u = np.zeros(n_dof)
    u[free_dofs] = u_free
    u[fixed_dofs] = 0.0
    r_fixed = K_sf @ u_free - P_fixed
    r = np.zeros(n_dof)
    r[fixed_dofs] = r_fixed
    r[free_dofs] = 0.0
    return (u, r)