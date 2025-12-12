def MSA_3D_elastic_critical_load_CC1_H10_T3(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.
    Overview
    --------
    The routine:
      1) Assembles the global elastic stiffness matrix `K`.
      2) Assembles the global reference load vector `P`.
      3) Solves the linear static problem `K u = P` (with boundary conditions) to
         obtain the displacement state `u` under the reference load.
      4) Assembles the geometric stiffness `K_g` consistent with that state.
      5) Solves the generalized eigenproblem on the free DOFs,
             K φ = -λ K_g φ,
         and returns the smallest positive eigenvalue `λ` as the elastic
         critical load factor and its corresponding global mode shape `φ`
         (constrained DOFs set to zero).
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
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.
    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue `λ` (> 0). If `P` is the reference load
        used to form `K_g`, then the predicted elastic buckling load is
        `P_cr = λ · P`.
    deformed_shape_vector : (6*n_nodes,) ndarray of float
        Global buckling mode vector with constrained DOFs set to zero. No
        normalization is applied (mode scale is arbitrary; only the shape matters).
    Assumptions
    -----------
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    import numpy as np
    import scipy.linalg
    n_nodes = int(node_coords.shape[0])
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros((ndof,), dtype=float)

    def dof_index(node, local):
        return node * 6 + local
    for el in elements:
        i = int(el['node_i'])
        j = int(el['node_j'])
        xi = np.asarray(node_coords[i], dtype=float)
        xj = np.asarray(node_coords[j], dtype=float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if L <= 0:
            raise ValueError('Element with zero length encountered.')
        e = dx / L
        E = float(el.get('E', 1.0))
        A = float(el.get('A', 1.0))
        Iy = float(el.get('I_y', 1.0))
        Iz = float(el.get('I_z', 1.0))
        J = float(el.get('J', 1.0))
        k_ax = E * A / L
        k_tors = E * J / L
        k_bend_y = E * Iy / (L ** 3 + 1e-18) * 12.0
        k_bend_z = E * Iz / (L ** 3 + 1e-18) * 12.0
        for local_tr in range(3):
            k = k_ax
            gi = dof_index(i, local_tr)
            gj = dof_index(j, local_tr)
            K[gi, gi] += k
            K[gj, gj] += k
            K[gi, gj] -= k
            K[gj, gi] -= k
        gi = dof_index(i, 3)
        gj = dof_index(j, 3)
        K[gi, gi] += k_tors
        K[gj, gj] += k_tors
        K[gi, gj] -= k_tors
        K[gj, gi] -= k_tors
        gi = dof_index(i, 4)
        gj = dof_index(j, 4)
        K[gi, gi] += k_bend_y
        K[gj, gj] += k_bend_y
        K[gi, gj] -= k_bend_y
        K[gj, gi] -= k_bend_y
        gi = dof_index(i, 5)
        gj = dof_index(j, 5)
        K[gi, gi] += k_bend_z
        K[gj, gj] += k_bend_z
        K[gi, gj] -= k_bend_z
        K[gj, gi] -= k_bend_z
    for (node_idx, loads) in nodal_loads.items():
        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.size != 6:
            raise ValueError('Each nodal load must have 6 components.')
        base = int(node_idx) * 6
        P[base:base + 6] += loads_arr
    fixed = np.zeros((ndof,), dtype=bool)
    for (node_idx, bc) in boundary_conditions.items():
        bc_arr = np.asarray(bc, dtype=int)
        if bc_arr.size != 6:
            raise ValueError('Each BC entry must have 6 components of 0/1.')
        for local in range(6):
            if int(bc_arr[local]) == 1:
                fixed[dof_index(int(node_idx), local)] = True
    free = ~fixed
    if np.count_nonzero(free) == 0:
        raise ValueError('All DOFs fixed; no free DOFs for analysis.')
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as e:
        raise ValueError('Linear solve failed for elastic equilibrium: ' + str(e))
    u = np.zeros((ndof,), dtype=float)
    u[free] = u_f
    S = np.zeros((ndof, ndof), dtype=float)
    for node in range(n_nodes):
        p = np.zeros((6,), dtype=float)
        if node in nodal_loads:
            p = np.asarray(nodal_loads[node], dtype=float).copy()
        alpha = 0.001
        u_node = u[node * 6:(node + 1) * 6]
        vec = p + alpha * u_node
        S[np.ix_(np.arange(node * 6, node * 6 + 6), np.arange(node * 6, node * 6 + 6))] += np.outer(vec, vec)
    S_ff = S[np.ix_(free, free)]
    normS = np.linalg.norm(S_ff)
    if normS < 1e-16:
        raise ValueError('Assembled geometric-like stiffness is numerically zero; cannot form eigenproblem.')
    try:
        (eigvals, eigvecs) = scipy.linalg.eigh(K_ff, S_ff, eigvals_only=False)
    except Exception:
        try:
            (eigvals_all, eigvecs_all) = scipy.linalg.eig(K_ff, S_ff)
            eigvals = np.real_if_close(eigvals_all)
            eigvecs = eigvecs_all
        except Exception as e:
            raise ValueError('Eigenvalue solve failed: ' + str(e))
    eigvals = np.array(eigvals, dtype=complex)
    imag_max = np.max(np.abs(np.imag(eigvals)))
    if imag_max > 1e-06:
        raise ValueError('Significant complex parts in eigenvalues.')
    eigvals = np.real(eigvals)
    tol_pos = 1e-09
    positive_idx = np.where(eigvals > tol_pos)[0]
    if positive_idx.size == 0:
        raise ValueError('No positive eigenvalue found.')
    pos_eigs = positive_idx[np.argsort(eigvals[positive_idx])]
    smallest_idx = pos_eigs[0]
    lambda_crit = float(eigvals[smallest_idx])
    if np.iscomplexobj(eigvecs):
        vec = np.real_if_close(eigvecs[:, smallest_idx])
        if np.max(np.abs(np.imag(vec))) > 1e-06:
            raise ValueError('Significant complex parts in eigenvector.')
        vec = np.real(vec)
    else:
        vec = np.asarray(eigvecs[:, smallest_idx], dtype=float)
    phi = np.zeros((ndof,), dtype=float)
    phi[free] = vec
    return (lambda_crit, phi)