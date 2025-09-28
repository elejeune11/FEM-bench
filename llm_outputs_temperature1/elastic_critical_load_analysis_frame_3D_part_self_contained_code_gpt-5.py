def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    import scipy
    from typing import Sequence as _Sequence
    import pytest as _pytest
    n_nodes = int(node_coords.shape[0])
    if node_coords.shape[1] != 3:
        raise ValueError('node_coords must have shape (n_nodes, 3)')
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    K_g = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    for (ni, loads) in nodal_loads.items():
        if ni < 0 or ni >= n_nodes:
            raise ValueError('nodal_loads contains invalid node index')
        arr = np.asarray(loads, dtype=float).reshape(-1)
        if arr.size != 6:
            raise ValueError('Each nodal load must be length 6')
        P[ni * 6:ni * 6 + 6] = arr
    constrained = np.zeros(ndof, dtype=bool)
    for (ni, spec) in boundary_conditions.items():
        if ni < 0 or ni >= n_nodes:
            raise ValueError('boundary_conditions contains invalid node index')
        spec_list = list(spec)
        if len(spec_list) == 6 and all((isinstance(x, (bool, np.bool_)) for x in spec_list)):
            mask = np.array(spec_list, dtype=bool)
            constrained[ni * 6:ni * 6 + 6] = mask
        else:
            for idx in spec_list:
                ii = int(idx)
                if ii < 0 or ii >= 6:
                    raise ValueError('Boundary condition DOF index must be in [0..5]')
                constrained[ni * 6 + ii] = True
    free = np.where(~constrained)[0]
    if free.size == 0:
        raise ValueError('All DOFs are constrained; no free DOFs remain.')

    def _element_transform(xi, xj, local_z):
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError('Element has zero or invalid length.')
        ex = dx / L
        if local_z is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ex, ref)) > 0.999:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            ez_guess = ref
        else:
            ez_guess = np.asarray(local_z, dtype=float).reshape(3)
            nz = float(np.linalg.norm(ez_guess))
            if nz == 0.0 or not np.isfinite(nz):
                raise ValueError('Provided local_z is invalid.')
            ez_guess = ez_guess / nz
            if abs(np.dot(ex, ez_guess)) > 0.999:
                ref = np.array([0.0, 1.0, 0.0], dtype=float) if abs(ex[2]) > 0.9 else np.array([0.0, 0.0, 1.0], dtype=float)
                ez_guess = ref
        ey = np.cross(ez_guess, ex)
        ny = float(np.linalg.norm(ey))
        if ny < 1e-12:
            ref = np.array([0.0, 1.0, 0.0], dtype=float) if abs(ex[2]) > 0.9 else np.array([0.0, 0.0, 1.0], dtype=float)
            ey = np.cross(ref, ex)
            ny = float(np.linalg.norm(ey))
            if ny < 1e-12:
                raise ValueError('Failed to construct a valid local triad.')
        ey = ey / ny
        ez = np.cross(ex, ey)
        D = np.vstack((ex, ey, ez))
        T = np.zeros((12, 12), dtype=float)
        for k in range(4):
            T[k * 3:(k + 1) * 3, k * 3:(k + 1) * 3] = D
        return (L, T)
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        if ni < 0 or nj < 0 or ni >= n_nodes or (nj >= n_nodes):
            raise ValueError('Element references invalid node indices.')
        xi = np.asarray(node_coords[ni], dtype=float)
        xj = np.asarray(node_coords[nj], dtype=float)
        (L, T) = _element_transform(xi, xj, elem.get('local_z', None))
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['Iy'])
        Iz = float(elem['Iz'])
        J = float(elem['J'])
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        k_glob = T.T @ k_loc @ T
        dofs = np.r_[np.arange(ni * 6, ni * 6 + 6), np.arange(nj * 6, nj * 6 + 6)]
        K[np.ix_(dofs, dofs)] += k_glob
    K_ff = K[np.ix_(free, free)]
    P_f = P[free]
    try:
        cond_K = np.linalg.cond(K_ff)
        if not np.isfinite(cond_K) or cond_K > 1000000000000.0:
            raise ValueError('Reduced stiffness matrix is ill-conditioned or singular.')
    except np.linalg.LinAlgError:
        raise ValueError('Failed to assess conditioning of reduced stiffness matrix.')
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        try:
            u_f = scipy.linalg.solve(K_ff, P_f, assume_a='sym', check_finite=True)
        except Exception:
            raise ValueError('Failed to solve for static displacement state.') from e
    u = np.zeros(ndof, dtype=float)
    u[free] = u_f
    any_geom = False
    for elem in elements:
        ni = int(elem['node_i'])
        nj = int(elem['node_j'])
        xi = np.asarray(node_coords[ni], dtype=float)
        xj = np.asarray(node_coords[nj], dtype=float)
        (L, T) = _element_transform(xi, xj, elem.get('local_z', None))
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['Iy'])
        Iz = float(elem['Iz'])
        J = float(elem['J'])
        I_rho = float(elem['I_rho'])
        k_loc = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        dofs = np.r_[np.arange(ni * 6, ni * 6 + 6), np.arange(nj * 6, nj * 6 + 6)]
        u_e_g = u[dofs]
        u_e_l = T @ u_e_g
        f_loc = k_loc @ u_e_l
        Fx2 = float(f_loc[6])
        Mx2 = float(f_loc[9])
        My1 = float(f_loc[4])
        Mz1 = float(f_loc[5])
        My2 = float(f_loc[10])
        Mz2 = float(f_loc[11])
        k_g_loc = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        if np.linalg.norm(k_g_loc, ord='fro') > 0.0:
            any_geom = True
        k_g_glob = T.T @ k_g_loc @ T
        K_g[np.ix_(dofs, dofs)] += k_g_glob
    if not any_geom or np.linalg.norm(K_g[np.ix_(free, free)], ord='fro') == 0.0:
        raise ValueError('Geometric stiffness is zero under the provided reference load state.')
    K_g_ff = K_g[np.ix_(free, free)]
    try:
        (eigvals, eigvecs) = scipy.linalg.eig(K_ff, -K_g_ff, right=True, check_finite=True)
    except Exception as e:
        raise ValueError('Generalized eigenvalue problem failed.') from e
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    imag_tol = 1e-06
    pos_tol = 1e-12
    valid = []
    for (i, lam) in enumerate(eigvals):
        lam_r = float(np.real(lam))
        lam_i = float(np.imag(lam))
        if abs(lam_i) <= imag_tol * max(1.0, abs(lam_r)):
            if lam_r > pos_tol:
                valid.append((lam_r, i))
    if not valid:
        raise ValueError('No positive real eigenvalues found for buckling problem.')
    valid.sort(key=lambda x: x[0])
    (lambda_min, idx_min) = valid[0]
    phi_f = np.real(eigvecs[:, idx_min])
    phi = np.zeros(ndof, dtype=float)
    phi[free] = phi_f
    return (float(lambda_min), phi)