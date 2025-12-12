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
    """
    import numpy as np
    import scipy
    from scipy import linalg
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be (N,3) array')
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    for (nidx, loads) in (nodal_loads or {}).items():
        if not 0 <= nidx < n_nodes:
            raise ValueError('nodal_loads contains invalid node index')
        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.size != 6:
            raise ValueError('nodal_loads entries must have length 6')
        P[6 * nidx:6 * nidx + 6] = loads_arr
    fixed = np.zeros(ndof, dtype=bool)
    for (nidx, bcvals) in (boundary_conditions or {}).items():
        if not 0 <= nidx < n_nodes:
            raise ValueError('boundary_conditions contains invalid node index')
        arr = np.asarray(bcvals, dtype=int)
        if arr.size != 6:
            raise ValueError('boundary_conditions for each node must be length 6')
        for k in range(6):
            v = int(arr[k])
            if v not in (0, 1):
                raise ValueError('boundary_conditions values must be 0 or 1')
            fixed[6 * nidx + k] = v == 1

    def element_matrices(elem):
        node_i = int(elem['node_i'])
        node_j = int(elem['node_j'])
        if not (0 <= node_i < n_nodes and 0 <= node_j < n_nodes):
            raise ValueError('element node index out of range')
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        I_y = float(elem['I_y'])
        I_z = float(elem['I_z'])
        J = float(elem['J'])
        xi = node_coords[node_i]
        xj = node_coords[node_j]
        v = xj - xi
        L = np.linalg.norm(v)
        if L <= 0.0:
            raise ValueError('Element length must be positive')
        x_local = v / L
        provided_z = elem.get('local_z', None)
        if provided_z is None:
            cand = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(cand, x_local)) > 0.999:
                cand = np.array([0.0, 1.0, 0.0], dtype=float)
            local_z_raw = cand
        else:
            local_z_raw = np.asarray(provided_z, dtype=float)
            if local_z_raw.shape != (3,):
                raise ValueError('local_z must be length-3 vector')
            if np.linalg.norm(local_z_raw) == 0.0:
                raise ValueError('local_z must be non-zero')
            if abs(np.dot(local_z_raw / np.linalg.norm(local_z_raw), x_local)) > 0.999:
                cand = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(np.dot(cand, x_local)) > 0.999:
                    cand = np.array([0.0, 1.0, 0.0], dtype=float)
                local_z_raw = cand
        proj = np.dot(local_z_raw, x_local) * x_local
        z_temp = local_z_raw - proj
        znorm = np.linalg.norm(z_temp)
        if znorm < 1e-12:
            cand = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(cand, x_local)) > 0.999:
                cand = np.array([0.0, 1.0, 0.0], dtype=float)
            proj = np.dot(cand, x_local) * x_local
            z_temp = cand - proj
            znorm = np.linalg.norm(z_temp)
            if znorm < 1e-12:
                raise ValueError('Cannot determine local z axis for element')
        z_local = z_temp / znorm
        y_local = np.cross(z_local, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        R = np.column_stack((x_local, y_local, z_local))
        B = np.zeros((12, 12), dtype=float)
        for k in range(4):
            B[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        k_local = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k_local[0, 0] = k_ax
        k_local[0, 6] = -k_ax
        k_local[6, 0] = -k_ax
        k_local[6, 6] = k_ax
        G = E / (2.0 * (1.0 + nu))
        k_tor = G * J / L
        k_local[3, 3] = k_tor
        k_local[3, 9] = -k_tor
        k_local[9, 3] = -k_tor
        k_local[9, 9] = k_tor
        k_bz = E * I_z / L ** 3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k_local[idx_bz[a], idx_bz[b]] += k_bz[a, b]
        k_by = E * I_y / L ** 3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k_local[idx_by[a], idx_by[b]] += k_by[a, b]
        return {'node_i': node_i, 'node_j': node_j, 'L': L, 'B': B, 'k_local': k_local, 'k_global_template_indices': None, 'R': R}
    elem_data = []
    for elem in elements:
        data = element_matrices(elem)
        k_local = data['k_local']
        B = data['B']
        k_global = B @ k_local @ B.T
        node_i = data['node_i']
        node_j = data['node_j']
        ind = np.concatenate((np.arange(6 * node_i, 6 * node_i + 6), np.arange(6 * node_j, 6 * node_j + 6)))
        K[np.ix_(ind, ind)] += k_global
        data['ind'] = ind
        elem_data.append({'node_i': node_i, 'node_j': node_j, 'k_local': data['k_local'], 'B': data['B'], 'L': data['L'], 'ind': ind, 'E': float(elem['E']), 'nu': float(elem['nu']), 'A': float(elem['A']), 'I_y': float(elem['I_y']), 'I_z': float(elem['I_z']), 'J': float(elem['J'])})
    free_idx = np.where(~fixed)[0]
    if free_idx.size == 0:
        raise ValueError('No free degrees of freedom')
    K_ff = K[np.ix_(free_idx, free_idx)]
    P_f = P[free_idx]
    try:
        u_f = linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as e:
        raise ValueError('Static solve failed: ' + str(e))
    u = np.zeros(ndof, dtype=float)
    u[free_idx] = u_f
    u[fixed] = 0.0
    K_g = np.zeros((ndof, ndof), dtype=float)
    for ed in elem_data:
        ind = ed['ind']
        B = ed['B']
        k_local = ed['k_local']
        L = ed['L']
        u_e = u[ind]
        d_local = B.T @ u_e
        f_local = k_local @ d_local
        N_axial = float(f_local[0])
        kgeo4 = N_axial / (30.0 * L) * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]], dtype=float)
        k_g_local = np.zeros((12, 12), dtype=float)
        idx_bz = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k_g_local[idx_bz[a], idx_bz[b]] += kgeo4[a, b]
        idx_by = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k_g_local[idx_by[a], idx_by[b]] += kgeo4[a, b]
        k_g_global = B @ k_g_local @ B.T
        K_g[np.ix_(ind, ind)] += k_g_global
    K_g_ff = K_g[np.ix_(free_idx, free_idx)]
    A = K_ff
    Bmat = -K_g_ff
    try:
        (eigvals, eigvecs) = linalg.eig(A, Bmat)
    except Exception as e:
        raise ValueError('Eigenproblem solve failed: ' + str(e))
    imag_tol = 1e-08
    real_mask = np.abs(eigvals.imag) < imag_tol
    if not np.any(real_mask):
        raise ValueError('No sufficiently real eigenvalues found')
    real_eigvals = eigvals[real_mask].real
    real_eigvecs = eigvecs[:, real_mask]
    pos_tol = 1e-12
    pos_mask = real_eigvals > pos_tol
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalue found')
    pos_vals = real_eigvals[pos_mask]
    pos_vecs = real_eigvecs[:, pos_mask]
    idx_min = np.argmin(pos_vals)
    lambda_min = float(pos_vals[idx_min])
    phi_free = pos_vecs[:, idx_min].real
    phi = np.zeros(ndof, dtype=float)
    phi[free_idx] = phi_free
    phi[fixed] = 0.0
    return (lambda_min, phi)