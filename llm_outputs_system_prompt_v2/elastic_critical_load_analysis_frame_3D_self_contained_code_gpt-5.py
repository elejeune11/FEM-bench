def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy
    from typing import Sequence
    n_nodes = int(node_coords.shape[0])
    dof_per_node = 6
    total_dofs = n_nodes * dof_per_node

    def _node_dofs(n):
        base = n * dof_per_node
        return np.arange(base, base + dof_per_node, dtype=int)
    constrained = np.zeros(total_dofs, dtype=bool)
    for (n, spec) in (boundary_conditions or {}).items():
        if n < 0 or n >= n_nodes:
            raise ValueError('Boundary condition references invalid node index.')
        dofs = _node_dofs(n)
        spec_list = list(spec)
        if all((isinstance(x, bool) for x in spec_list)):
            if len(spec_list) != 6:
                raise ValueError('Boolean BC specification must have length 6.')
            constrained[dofs] = np.array(spec_list, dtype=bool)
        elif all((isinstance(x, (int, np.integer)) for x in spec_list)):
            for idx in spec_list:
                if idx < 0 or idx >= 6:
                    raise ValueError('BC DOF index must be in 0..5.')
                constrained[dofs[idx]] = True
        else:
            raise ValueError('BC specification must be all bools or all integers.')
    free = np.where(~constrained)[0]
    if free.size == 0:
        raise ValueError('All DOFs are constrained.')
    I_k = []
    J_k = []
    V_k = []

    def _element_transformation(xi, xj, local_z):
        v = xj - xi
        L = float(np.linalg.norm(v))
        if L <= 0.0 or not np.isfinite(L):
            raise ValueError('Zero or invalid element length.')
        ex = v / L
        if local_z is None:
            zref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ex, zref)) > 0.99:
                zref = np.array([0.0, 1.0, 0.0])
        else:
            zref = np.array(local_z, dtype=float).reshape(3)
            if not np.isfinite(zref).all():
                raise ValueError('local_z contains invalid values.')
        ey = np.cross(zref, ex)
        ny = np.linalg.norm(ey)
        if ny < 1e-12:
            zref = np.array([0.0, 1.0, 0.0])
            ey = np.cross(zref, ex)
            ny = np.linalg.norm(ey)
            if ny < 1e-12:
                zref = np.array([1.0, 0.0, 0.0])
                ey = np.cross(zref, ex)
                ny = np.linalg.norm(ey)
                if ny < 1e-12:
                    raise ValueError('Cannot construct local triad.')
        ey = ey / ny
        ez = np.cross(ex, ey)
        R = np.column_stack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        for b in range(4):
            T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
        return (T, L, R)

    def _elastic_local_stiffness(E, G, A, Iy, Iz, J, L):
        k = np.zeros((12, 12), dtype=float)
        a = E * A / L
        k[0, 0] = a
        k[0, 6] = -a
        k[6, 0] = -a
        k[6, 6] = a
        t = G * J / L
        k[3, 3] = t
        k[3, 9] = -t
        k[9, 3] = -t
        k[9, 9] = t
        c = E * Iz
        k[1, 1] = 12 * c / L ** 3
        k[1, 5] = 6 * c / L ** 2
        k[1, 7] = -12 * c / L ** 3
        k[1, 11] = 6 * c / L ** 2
        k[5, 1] = 6 * c / L ** 2
        k[5, 5] = 4 * c / L
        k[5, 7] = -6 * c / L ** 2
        k[5, 11] = 2 * c / L
        k[7, 1] = -12 * c / L ** 3
        k[7, 5] = -6 * c / L ** 2
        k[7, 7] = 12 * c / L ** 3
        k[7, 11] = -6 * c / L ** 2
        k[11, 1] = 6 * c / L ** 2
        k[11, 5] = 2 * c / L
        k[11, 7] = -6 * c / L ** 2
        k[11, 11] = 4 * c / L
        d = E * Iy
        k[2, 2] = 12 * d / L ** 3
        k[2, 4] = -6 * d / L ** 2
        k[2, 8] = -12 * d / L ** 3
        k[2, 10] = -6 * d / L ** 2
        k[4, 2] = -6 * d / L ** 2
        k[4, 4] = 4 * d / L
        k[4, 8] = 6 * d / L ** 2
        k[4, 10] = 2 * d / L
        k[8, 2] = -12 * d / L ** 3
        k[8, 4] = 6 * d / L ** 2
        k[8, 8] = 12 * d / L ** 3
        k[8, 10] = 6 * d / L ** 2
        k[10, 2] = -6 * d / L ** 2
        k[10, 4] = 2 * d / L
        k[10, 8] = 6 * d / L ** 2
        k[10, 10] = 4 * d / L
        return k
    elem_data = []
    for e in elements:
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        xi = np.array(node_coords[ni], dtype=float)
        xj = np.array(node_coords[nj], dtype=float)
        local_z = e.get('local_z', None)
        (T, L, R) = _element_transformation(xi, xj, local_z)
        E = float(e['E'])
        nu = float(e.get('nu', 0.3))
        G = E / (2.0 * (1.0 + nu))
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        k_loc = _elastic_local_stiffness(E, G, A, Iy, Iz, J, L)
        k_glb = T.T @ k_loc @ T
        edofs = np.concatenate((_node_dofs(ni), _node_dofs(nj)))
        for a in range(12):
            ia = edofs[a]
            for b in range(12):
                ib = edofs[b]
                val = k_glb[a, b]
                if val != 0.0:
                    I_k.append(ia)
                    J_k.append(ib)
                    V_k.append(val)
        elem_data.append((ni, nj, edofs, T, L, k_loc))
    if len(V_k) == 0:
        raise ValueError('No stiffness assembled; check elements.')
    K = scipy.sparse.csr_matrix((V_k, (I_k, J_k)), shape=(total_dofs, total_dofs))
    P = np.zeros(total_dofs, dtype=float)
    if nodal_loads:
        for (n, load) in nodal_loads.items():
            if n < 0 or n >= n_nodes:
                raise ValueError('Load references invalid node index.')
            load_vec = np.array(load, dtype=float).reshape(-1)
            if load_vec.size != 6:
                raise ValueError('Nodal load vector must have length 6.')
            P[_node_dofs(n)] += load_vec
    K_ff = K[free][:, free]
    P_f = P[free]
    try:
        u_f = scipy.sparse.linalg.spsolve(K_ff, P_f)
    except Exception as err:
        K_ff_d = K_ff.toarray()
        try:
            u_f = scipy.linalg.solve(K_ff_d, P_f, assume_a='sym')
        except Exception:
            raise ValueError('Failed to solve static reference problem.') from err
    u = np.zeros(total_dofs, dtype=float)
    u[free] = u_f
    I_kg = []
    J_kg = []
    V_kg = []
    any_nonzero_kg = False
    for (ni, nj, edofs, T, L, k_loc) in elem_data:
        ue = u[edofs]
        u_local = T @ ue
        f_local = k_loc @ u_local
        N_tension_positive = f_local[0]
        N_comp = -N_tension_positive
        if not np.isfinite(N_comp):
            continue
        Kg_loc = np.zeros((12, 12), dtype=float)
        if abs(N_comp) > 0.0 and L > 0.0:
            S = N_comp / (30.0 * L)
            L1 = L
            L2 = L * L
            M = np.array([[36.0, 3.0 * L1, -36.0, 3.0 * L1], [3.0 * L1, 4.0 * L2, -3.0 * L1, -1.0 * L2], [-36.0, -3.0 * L1, 36.0, -3.0 * L1], [3.0 * L1, -1.0 * L2, -3.0 * L1, 4.0 * L2]], dtype=float)
            idx_y = [1, 5, 7, 11]
            idx_z = [2, 4, 8, 10]
            Kg_loc[np.ix_(idx_y, idx_y)] += S * M
            Kg_loc[np.ix_(idx_z, idx_z)] += S * M
            any_nonzero_kg = True
        Kg_glb = T.T @ Kg_loc @ T
        for a in range(12):
            ia = edofs[a]
            for b in range(12):
                ib = edofs[b]
                val = Kg_glb[a, b]
                if val != 0.0:
                    I_kg.append(ia)
                    J_kg.append(ib)
                    V_kg.append(val)
    if not any_nonzero_kg or len(V_kg) == 0:
        raise ValueError('Geometric stiffness is zero; reference state produces no axial compression.')
    Kg = scipy.sparse.csr_matrix((V_kg, (I_kg, J_kg)), shape=(total_dofs, total_dofs))
    K_ff = K[free][:, free].toarray()
    Kg_ff = Kg[free][:, free].toarray()
    A = K_ff
    B = -Kg_ff
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError('Non-finite entries in reduced matrices.')
    if np.linalg.norm(B, ord='fro') == 0.0:
        raise ValueError('Reduced geometric stiffness is zero.')
    try:
        (w, V) = scipy.linalg.eig(A, B)
    except Exception as err:
        raise ValueError('Generalized eigenvalue solve failed.') from err
    w = np.array(w)
    V = np.array(V)
    tol_imag = 1e-08
    real_mask = np.abs(w.imag) <= tol_imag * np.maximum(1.0, np.abs(w.real))
    w_real = w.real[real_mask]
    V_real = V[:, real_mask]
    if w_real.size == 0:
        raise ValueError('No real eigenvalues found.')
    tol_pos = 1e-10
    pos_mask = w_real > tol_pos
    if not np.any(pos_mask):
        raise ValueError('No positive eigenvalues found.')
    w_pos = w_real[pos_mask]
    V_pos = V_real[:, pos_mask]
    idx_min = int(np.argmin(w_pos))
    lambda_min = float(w_pos[idx_min])
    mode_free = V_pos[:, idx_min].real
    phi_full = np.zeros(total_dofs, dtype=float)
    phi_full[free] = mode_free
    return (lambda_min, phi_full)