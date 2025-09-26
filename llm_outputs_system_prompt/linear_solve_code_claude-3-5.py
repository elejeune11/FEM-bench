def linear_solve(P_global, K_global, fixed, free):
    n_dof = len(P_global)
    u = np.zeros(n_dof)
    nodal_reaction_vector = np.zeros(n_dof)
    K_ff = K_global[np.ix_(free, free)]
    K_fs = K_global[np.ix_(free, fixed)]
    K_sf = K_global[np.ix_(fixed, free)]
    P_f = P_global[free]
    cond_number = np.linalg.cond(K_ff)
    if cond_number >= 1e+16:
        raise ValueError(f'System is ill-conditioned. Condition number: {cond_number}')
    u_f = np.linalg.solve(K_ff, P_f)
    u[free] = u_f
    nodal_reaction_vector[fixed] = K_sf @ u_f
    return (u, nodal_reaction_vector)