def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy
    from typing import Optional, Sequence
    import pytest

    def _validate_and_prepare():
        if not isinstance(node_coords, np.ndarray) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
            raise ValueError('node_coords must be an ndarray of shape (n_nodes, 3).')
        n_nodes = node_coords.shape[0]
        ndof = 6 * n_nodes
        required_keys = {'node_i', 'node_j', 'E', 'nu', 'A', 'Iy', 'Iz', 'J', 'I_rho', 'local_z'}
        for e in elements:
            if not isinstance(e, dict):
                raise ValueError('Each element must be a dictionary with required keys.')
            missing = required_keys - set(e.keys())
            if missing:
                raise ValueError(f'Element missing keys: {missing}')
            ni = e['node_i']
            nj = e['node_j']
            if not (0 <= ni < n_nodes and 0 <= nj < n_nodes and (ni != nj)):
                raise ValueError('Element node indices out of range or identical.')
            pi = node_coords[ni]
            pj = node_coords[nj]
            L = np.linalg.norm(pj - pi)
            if not np.isfinite(L) or L <= 0.0:
                raise ValueError('Element length must be positive and finite.')
            if e['A'] <= 0 or e['E'] <= 0 or e['Iy'] <= 0 or (e['Iz'] <= 0) or (e['J'] <= 0):
                raise ValueError('Section and material properties must be positive.')
        is_fixed = np.zeros(ndof, dtype=bool)
        for (n, bc) in (boundary_conditions or {}).items():
            if not 0 <= n < n_nodes:
                raise ValueError('Boundary condition node index out of range.')
            if bc is None:
                continue
            if len(bc) == 6 and all((isinstance(b, (bool, np.bool_)) for b in bc)):
                for k in range(6):
                    if bc[k]:
                        is_fixed[n * 6 + k] = True
            else:
                for k in bc:
                    if not 0 <= int(k) < 6:
                        raise ValueError('Boundary condition DOF index must be in 0..5.')
                    is_fixed[n * 6 + int(k)] = True
        P = np.zeros(ndof, dtype=float)
        for (n, load) in (nodal_loads or {}).items():
            if not 0 <= n < n_nodes:
                raise ValueError('Load node index out of range.')
            if load is None:
                continue
            if len(load) != 6:
                raise ValueError('Nodal load vector must have length 6.')
            P[n * 6:(n + 1) * 6] += np.asarray(load, dtype=float)
        return (n_nodes, ndof, is_fixed, P)

    def _element_transformation_and_length(e):
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        xi = node_coords[ni]
        xj = node_coords[nj]
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        ex = dx / L
        z_hint = e.get('local_z', None)
        if z_hint is not None:
            z_hint = np.asarray(z_hint, dtype=float).reshape(3)
            if not np.all(np.isfinite(z_hint)):
                z_hint = None
        if z_hint is None or np.linalg.norm(z_hint) < 1e-12 or abs(np.dot(z_hint / np.linalg.norm(z_hint), ex)) > 0.999:
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(up, ex)) > 0.999:
                up = np.array([0.0, 1.0, 0.0])
            ey = np.cross(up, ex)
            ny = np.linalg.norm(ey)
            if ny < 1e-12:
                up = np.array([1.0, 0.0, 0.0])
                ey = np.cross(up, ex)
                ny = np.linalg.norm(ey)
                if ny < 1e-12:
                    raise ValueError('Failed to construct local axes.')
            ey = ey / ny
            ez = np.cross(ex, ey)
        else:
            zt = z_hint - np.dot(z_hint, ex) * ex
            nz = np.linalg.norm(zt)
            if nz < 1e-12:
                up = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(up, ex)) > 0.999:
                    up = np.array([0.0, 1.0, 0.0])
                ey = np.cross(up, ex)
                ey = ey / np.linalg.norm(ey)
                ez = np.cross(ex, ey)
            else:
                ez = zt / nz
                ey = np.cross(ez, ex)
                ey = ey / np.linalg.norm(ey)
                ez = np.cross(ex, ey)
        R = np.vstack([ex, ey, ez])
        return (L, R)

    def _local_elastic_stiffness(e, L):
        E = float(e['E'])
        A = float(e['A'])
        Iy = float(e['Iy'])
        Iz = float(e['Iz'])
        J = float(e['J'])
        nu = float(e['nu'])
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12), dtype=float)
        k_ax = E * A / L
        k[0, 0] += k_ax
        k[0, 6] -= k_ax
        k[6, 0] -= k_ax
        k[6, 6] += k_ax
        k_t = G * J / L
        k[3, 3] += k_t
        k[3, 9] -= k_t
        k[9, 3] -= k_t
        k[9, 9] += k_t
        EI_z = E * Iz
        L2 = L * L
        L3 = L2 * L
        kb = np.array([[12.0 * EI_z / L3, 6.0 * EI_z / L2, -12.0 * EI_z / L3, 6.0 * EI_z / L2], [6.0 * EI_z / L2, 4.0 * EI_z / L, -6.0 * EI_z / L2, 2.0 * EI_z / L], [-12.0 * EI_z / L3, -6.0 * EI_z / L2, 12.0 * EI_z / L3, -6.0 * EI_z / L2], [6.0 * EI_z / L2, 2.0 * EI_z / L, -6.0 * EI_z / L2, 4.0 * EI_z / L]], dtype=float)
        iy = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                k[iy[a], iy[b]] += kb[a, b]
        EI_y = E * Iy
        kb2 = np.array([[12.0 * EI_y / L3, 6.0 * EI_y / L2, -12.0 * EI_y / L3, 6.0 * EI_y / L2], [6.0 * EI_y / L2, 4.0 * EI_y / L, -6.0 * EI_y / L2, 2.0 * EI_y / L], [-12.0 * EI_y / L3, -6.0 * EI_y / L2, 12.0 * EI_y / L3, -6.0 * EI_y / L2], [6.0 * EI_y / L2, 2.0 * EI_y / L, -6.0 * EI_y / L2, 4.0 * EI_y / L]], dtype=float)
        iz = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                k[iz[a], iz[b]] += kb2[a, b]
        return k

    def _local_geometric_stiffness(N, L):
        Kg = np.zeros((12, 12), dtype=float)
        c = N / (30.0 * L)
        L2 = L * L
        block = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float) * c
        idx_y = [1, 5, 7, 11]
        for a in range(4):
            for b in range(4):
                Kg[idx_y[a], idx_y[b]] += block[a, b]
        idx_z = [2, 4, 8, 10]
        for a in range(4):
            for b in range(4):
                Kg[idx_z[a], idx_z[b]] += block[a, b]
        return Kg

    def _element_dof_indices(ni, nj):
        edofs = []
        for n in (ni, nj):
            base = n * 6
            edofs.extend([base + i for i in range(6)])
        return edofs
    (n_nodes, ndof, is_fixed, P) = _validate_and_prepare()
    free = np.where(~is_fixed)[0]
    if free.size == 0:
        raise ValueError('All DOFs are constrained; free set is empty.')
    K = np.zeros((ndof, ndof), dtype=float)
    elem_cache = []
    for e in elements:
        (L, R) = _element_transformation_and_length(e)
        k_loc = _local_elastic_stiffness(e, L)
        T = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            T[blk * 3:(blk + 1) * 3, blk * 3:(blk + 1) * 3] = R
        k_gl = T.T @ k_loc @ T
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        edofs = _element_dof_indices(ni, nj)
        for (a_local, A) in enumerate(edofs):
            for (b_local, B) in enumerate(edofs):
                K[A, B] += k_gl[a_local, b_local]
        elem_cache.append((e, L, R))
    Kff = K[np.ix_(free, free)]
    Pf = P[free]
    try:
        uf = scipy.linalg.solve(Kff, Pf, assume_a='sym', check_finite=True, overwrite_a=False, overwrite_b=False)
    except Exception as err:
        raise ValueError(f'Failed to solve linear static problem on free DOFs: {err}')
    u = np.zeros(ndof, dtype=float)
    u[free] = uf
    Kg = np.zeros((ndof, ndof), dtype=float)
    any_nonzero_N = False
    for (e, L, R) in elem_cache:
        T = np.zeros((12, 12), dtype=float)
        for blk in range(4):
            T[blk * 3:(blk + 1) * 3, blk * 3:(blk + 1) * 3] = R
        ni = int(e['node_i'])
        nj = int(e['node_j'])
        edofs = _element_dof_indices(ni, nj)
        u_e = u[edofs]
        u_loc = T @ u_e
        k_loc = _local_elastic_stiffness(e, L)
        q_loc = k_loc @ u_loc
        N = 0.5 * (q_loc[0] - q_loc[6])
        if abs(N) > 1e-14:
            any_nonzero_N = True
        Kg_loc = _local_geometric_stiffness(N, L)
        Kg_gl = T.T @ Kg_loc @ T
        for (a_local, A) in enumerate(edofs):
            for (b_local, B) in enumerate(edofs):
                Kg[A, B] += Kg_gl[a_local, b_local]
    Kg_ff = Kg[np.ix_(free, free)]
    if not any_nonzero_N or np.linalg.norm(Kg_ff, ord='fro') <= 1e-14 * max(1.0, np.linalg.norm(Kff, ord='fro')):
        raise ValueError('Geometric stiffness is negligible or zero under the provided reference load state; no buckling can be predicted.')
    try:
        (evals, evecs) = scipy.linalg.eig(Kff, Kg_ff, check_finite=True)
    except Exception as err:
        raise ValueError(f'Generalized eigenvalue solve failed: {err}')
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    finite_mask = np.isfinite(evals.real) & np.isfinite(evals.imag)
    evals = evals[finite_mask]
    evecs = evecs[:, finite_mask]
    if evals.size == 0:
        raise ValueError('No finite eigenvalues returned.')
    mu_real = evals.real
    mu_imag = evals.imag
    rel = np.maximum(1.0, np.abs(mu_real))
    real_mask = np.abs(mu_imag) <= 1e-08 * rel
    mu_real = mu_real[real_mask]
    evecs = evecs[:, real_mask]
    if mu_real.size == 0:
        raise ValueError('No sufficiently real eigenvalues found.')
    lambdas = -mu_real
    pos_mask = lambdas > 1e-12
    if not np.any(pos_mask):
        raise ValueError('No positive elastic critical load factor found.')
    lambdas_pos = lambdas[pos_mask]
    evecs_pos = evecs[:, pos_mask]
    idx_min = int(np.argmin(lambdas_pos))
    lambda_cr = float(lambdas_pos[idx_min])
    phi_free = evecs_pos[:, idx_min].real
    phi = np.zeros(ndof, dtype=float)
    phi[free] = phi_free
    return (lambda_cr, phi)