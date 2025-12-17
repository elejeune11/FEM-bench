def MSA_3D_solve_linear_CC0_H1_T3(P_global: np.ndarray, K_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    import numpy as np
    dof_total = 6 * int(n_nodes)
    P = np.asarray(P_global, dtype=float).reshape(-1)
    K = np.asarray(K_global, dtype=float)
    fixed_mask = np.zeros(dof_total, dtype=bool)
    if boundary_conditions is not None:
        for node_idx, bc in boundary_conditions.items():
            try:
                i = int(node_idx)
            except Exception:
                continue
            if i < 0 or i >= n_nodes:
                continue
            bc_arr = np.asarray(bc, dtype=bool).reshape(-1)
            if bc_arr.size < 6:
                padded = np.zeros(6, dtype=bool)
                padded[:bc_arr.size] = bc_arr
                bc_arr = padded
            fixed_mask[i * 6:i * 6 + 6] = bc_arr[:6]
    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    fixed_idx = np.flatnonzero(fixed_mask)
    u = np.zeros(dof_total, dtype=float)
    r = np.zeros(dof_total, dtype=float)
    if free_idx.size == 0:
        if fixed_idx.size > 0:
            r[fixed_idx] = -P[fixed_idx]
        return (u, r)
    K_ff = K[np.ix_(free_idx, free_idx)]
    P_f = P[free_idx]
    cond_val = np.linalg.cond(K_ff)
    if not np.isfinite(cond_val) or cond_val >= 1e+16:
        raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned or singular.')
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        raise ValueError('Reduced stiffness matrix K_ff is ill-conditioned or singular.') from e
    u[free_idx] = u_f
    if fixed_idx.size > 0:
        K_cf = K[np.ix_(fixed_idx, free_idx)]
        P_c = P[fixed_idx]
        r_c = K_cf @ u_f - P_c
        r[fixed_idx] = r_c
    return (u, r)