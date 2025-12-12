def MSA_3D_solve_linear_CC0_H1_T1(P_global: np.ndarray, K_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    (fixed, free) = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    u = np.zeros_like(P_global)
    r = np.zeros_like(P_global)
    if len(free) > 0:
        K_ff = K_global[np.ix_(free, free)]
        P_f = P_global[free]
        cond = np.linalg.cond(K_ff)
        if cond >= 1e+16:
            raise ValueError(f'Reduced stiffness matrix K_ff is ill-conditioned (cond = {cond:.2e} ≥ 1e16).')
        u_free = np.linalg.solve(K_ff, P_f)
        u[free] = u_free
        if len(fixed) > 0:
            K_sf = K_global[np.ix_(fixed, free)]
            P_fixed = P_global[fixed]
            r_fixed = K_sf @ u_free - P_fixed
            r[fixed] = r_fixed
    elif len(fixed) > 0:
        K_sf = K_global[np.ix_(fixed, free)]
        P_fixed = P_global[fixed]
        r_fixed = -P_fixed
        r[fixed] = r_fixed
    return (u, r)