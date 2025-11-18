def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy
    from typing import Sequence
    if not isinstance(node_coords, np.ndarray) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be a (n_nodes, 3) ndarray.')
    n_nodes = node_coords.shape[0]
    if n_nodes < 2:
        raise ValueError('At least two nodes are required.')
    n_dof = 6 * n_nodes

    def build_local_axes_and_T(node_i: int, node_j: int, local_z_dir):
        xi = node_coords[node_i].astype(float)
        xj = node_coords[node_j].astype(float)
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element length must be positive and finite.')
        ex = dx / L
        if local_z_dir is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ref, ex)) > 0.99:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref = np.array(local_z_dir, dtype=float)
            nref = np.linalg.norm(ref)
            if not np.isfinite(nref) or nref == 0.0:
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
        ey = np.cross(ref, ex)
        ny = np.linalg.norm(ey)
        if ny < 1e-12:
            t = np.array([1.0, 0.0, 0.0], dtype=float) if abs(ex[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
            ey = np.cross(t, ex)
            ny = np.linalg.norm(ey)
            if ny < 1e-12:
                t = np.array([0.0, 0.0, 1.0], dtype=float)
                ey = np.cross(t, ex)
                ny = np.linalg.norm(ey)
                if ny < 1e-12:
                    raise ValueError('Failed to construct a valid local triad.')
        ey = ey / ny
        ez = np.cross(ex, ey)
        Lambda = np.vstack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        T[0:3, 0:3] = Lambda
        T[3:6, 3:6] = Lambda
        T[6:9, 6:9] = Lambda
        T[9:12, 9:12] = Lambda
        return (L, Lambda, T)

    def local_elastic_stiffness(E, G, A, Iy, Iz, J, L):
        k = np.zeros((12, 12), dtype=float)
        L2 = L * L
        L3 = L2 * L
        EA_L = E * A / L
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        GJ_L = G * J / L
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        c1 = 12.0 * E * Iz / L3
        c2 = 6.0 * E * Iz / L2
        c3 = 4.0 * E * Iz / L
        c4 = 2.0 * E * Iz / L
        idx = [1, 5, 7, 11]
        kbz = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]], dtype=float)
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += kbz[a, b]
        c1 = 12.0 * E * Iy / L3
        c2 = 6.0 * E * Iy / L2
        c3 = 4.0 * E * Iy / L
        c4 = 2.0 * E * Iy / L
        idx = [2, 4, 8, 10]
        kby = np.array([[c1, -c2, -c1, -c2], [-c2, c3, c2, c4], [-c1, c2, c1, c2], [-c2, c4, c2, c3]], dtype=float)
        for a in range(4):
            for b in range(4):
                k[idx[a], idx[b]] += kby[a, b]
        return k

    def local_geometric_stiffness(N, L):
        kg = np.zeros((12, 12), dtype=float)
        if N == 0.0:
            return kg
        c = N / (30.0 * L)
        M = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]], dtype=float)
        idx_v = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                kg[idx_v[a], idx_v[b]] += c * M[a, b]
        idx_w = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                kg[idx_w[a], idx_w[b]] += c * M[a, b]
        return kg
    K = np.zeros((n_dof, n_dof), dtype=float)
    elem_data = []
    for e in elements:
        try:
            i = int(e['node_i'])
            j = int(e['node_j'])
            E = float(e['E'])
            nu = float(e['nu'])
            A = float(e['A'])
            Iy = float(e['Iy'])
            Iz = float(e['Iz'])
            J = float(e['J'])
        except Exception as ex:
            raise ValueError(f'Element properties missing or invalid: {ex}')
        local_z_dir = e.get('local_z', None)
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError('Element node indices out of range.')
        (L, Lambda, T) = build_local_axes_and_T(i, j, local_z_dir)
        if L <= 0.0:
            raise ValueError('Element length must be positive.')
        if any((not np.isfinite(val) for val in (E, nu, A, Iy, Iz, J))):
            raise ValueError('Element properties must be finite numbers.')
        if A <= 0 or Iy <= 0 or Iz <= 0 or (J <= 0):
            raise ValueError('Section properties must be positive.')
        G = E / (2.0 * (1.0 + nu))
        k_local = local_elastic_stiffness(E, G, A, Iy, Iz, J, L)
        T_full = np.zeros((12, 12), dtype=float)
        T_full[0:12, 0:12] = T
        k_global = T_full.T @ k_local @ T_full
        idofs = np.array([6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], dtype=int)
        K[np.ix_(idofs, idofs)] += k_global
        elem_data.append((idofs, T_full, L, E, A))
    K = 0.5 * (K + K.T)
    P = np.zeros(n_dof, dtype=float)
    if nodal_loads:
        for (nd, load) in nodal_loads.items():
            if not 0 <= nd < n_nodes:
                raise ValueError('Load node index out of range.')
            vec = np.array(load, dtype=float).reshape(-1)
            if vec.size != 6:
                raise ValueError('Each nodal load must be a length-6 sequence.')
            idx = slice(6 * nd, 6 * nd + 6)
            P[idx] += vec
    constrained = np.zeros(n_dof, dtype=bool)
    if boundary_conditions:
        for (nd, spec) in boundary_conditions.items():
            if not 0 <= nd < n_nodes:
                raise ValueError('BC node index out of range.')
            spec_seq = list(spec)
            if len(spec_seq) == 6 and all((type(v) is bool or isinstance(v, np.bool_) for v in spec_seq)):
                mask = np.array(spec_seq, dtype=bool)
            else:
                mask = np.zeros(6, dtype=bool)
                for v in spec_seq:
                    iv = int(v)
                    if iv < 0 or iv > 5:
                        raise ValueError('BC index must be in 0..5.')
                    mask[iv] = True
            constrained[6 * nd:6 * nd + 6] = constrained[6 * nd:6 * nd + 6] | mask
    free = np.where(~constrained)[0]
    if free.size == 0:
        raise ValueError('No free DOFs remain after applying boundary conditions.')
    fixed = np.where(constrained)[0]
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym')
    except Exception as ex:
        raise ValueError(f'Failed to solve linear system (check BCs and stiffness): {ex}')
    u = np.zeros(n_dof, dtype=float)
    u[free] = u_f
    K_g = np.zeros((n_dof, n_dof), dtype=float)
    any_axial = False
    for (idofs, T_full, L, E, A) in elem_data:
        u_elem_global = u[idofs]
        d_local = T_full @ u_elem_global
        N = E * A / L * (d_local[6] - d_local[0])
        if N != 0.0 and np.isfinite(N):
            any_axial = True
            kg_local = local_geometric_stiffness(N, L)
            kg_global = T_full.T @ kg_local @ T_full
            K_g[np.ix_(idofs, idofs)] += kg_global
    K_g = 0.5 * (K_g + K_g.T)
    if not any_axial or np.allclose(K_g, 0.0):
        raise ValueError('Reference load state produced no axial forces for geometric stiffness; cannot perform buckling analysis.')
    K_g_ff = K_g[np.ix_(free, free)]
    try:
        C = -scipy.linalg.solve(K_ff, K_g_ff, assume_a='sym')
    except Exception as ex:
        raise ValueError(f'Failed to form reduced operator for eigenproblem: {ex}')
    try:
        (evals, evecs) = scipy.linalg.eig(C)
    except Exception as ex:
        raise ValueError(f'Failed to solve eigenproblem: {ex}')
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    imag_abs = np.abs(evals.imag)
    real_part = evals.real
    tol_imag = 1e-08
    tol_pos = 1e-10
    mask_real = imag_abs <= tol_imag * np.maximum(1.0, np.abs(real_part))
    real_evals = real_part[mask_real]
    real_evecs = evecs[:, mask_real]
    pos_mask = real_evals > tol_pos
    pos_evals = real_evals[pos_mask]
    pos_evecs = real_evecs[:, pos_mask]
    if pos_evals.size == 0:
        raise ValueError('No positive real eigenvalues found for buckling problem.')
    idx_min = int(np.argmin(pos_evals))
    lambda_cr = float(pos_evals[idx_min])
    mode_f = pos_evecs[:, idx_min].real
    mode_full = np.zeros(n_dof, dtype=float)
    mode_full[free] = mode_f
    return (lambda_cr, mode_full)