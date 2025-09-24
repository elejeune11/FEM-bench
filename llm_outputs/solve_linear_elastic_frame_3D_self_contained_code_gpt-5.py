def solve_linear_elastic_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    The function assembles the global stiffness matrix (K) and load vector (P),
    partitions degrees of freedom (DOFs) into free and fixed sets, solves the
    reduced system for displacements at the free DOFs, and computes true support
    reactions at the fixed DOFs.
    Coordinate system: global right-handed Cartesian. Element local axes follow the
    beam axis (local x) with orientation defined via a reference vector.
    Condition number policy: the system is solved only if the free–free stiffness
    submatrix K_ff is well-conditioned (cond(K_ff) < 1e16). Otherwise a ValueError
    is raised.
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
            'local_z' : (3,) array or None
                Optional unit vector to define the local z-direction for transformation
                matrix orientation (must be unit length and not parallel to the beam axis).
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index → 6-element iterable of 0 (free) or 1 (fixed). Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index → 6-element [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes ⇒ zero loads.
    Returns
    -------
    u : (6*N,) ndarray
        Global displacement vector ordered as [UX, UY, UZ, RX, RY, RZ] for each node
        in sequence. Values are computed at free DOFs; fixed DOFs are zero.
    r : (6*N,) ndarray
        Global reaction force/moment vector with nonzeros only at fixed DOFs.
        Reactions are computed as internal elastic forces minus applied loads at the
        fixed DOFs; free DOFs have zero entries.
    Raises
    ------
    ValueError
        If the free-free submatrix K_ff is ill-conditioned (cond(K_ff) ≥ 1e16).
    Notes
    -----
    """
    import numpy as np
    from typing import Sequence
    import pytest
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError('node_coords must be an (N,3) array.')
    N = node_coords.shape[0]
    ndof = 6 * N
    K = np.zeros((ndof, ndof), dtype=float)
    P = np.zeros(ndof, dtype=float)
    if nodal_loads is not None:
        for (n, loads) in nodal_loads.items():
            if n < 0 or n >= N:
                raise ValueError('nodal_loads contains an invalid node index.')
            loads_arr = np.asarray(loads, dtype=float).reshape(-1)
            if loads_arr.size != 6:
                raise ValueError('Each nodal load entry must have 6 components.')
            base = 6 * n
            P[base:base + 6] += loads_arr

    def build_rotation(xi: np.ndarray, xj: np.ndarray, local_z_vec):
        dx = xj - xi
        L = np.linalg.norm(dx)
        if L <= 0.0:
            raise ValueError('Element has zero length.')
        ex = dx / L
        if local_z_vec is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ref, ex)) > 0.999:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref = np.asarray(local_z_vec, dtype=float).reshape(3)
            nref = np.linalg.norm(ref)
            if nref == 0.0:
                raise ValueError('Provided local_z vector has zero length.')
            ref = ref / nref
        v = ref - np.dot(ref, ex) * ex
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            ref2 = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(ref2, ex)) > 0.999:
                ref2 = np.array([0.0, 1.0, 0.0], dtype=float)
            v = ref2 - np.dot(ref2, ex) * ex
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                raise ValueError('Cannot construct a valid local axis system; reference vector parallel to element axis.')
        ez = v / nv
        ey = np.cross(ez, ex)
        R = np.vstack((ex, ey, ez))
        return (R, L)
    for elem in elements:
        i = int(elem['node_i'])
        j = int(elem['node_j'])
        if i < 0 or i >= N or j < 0 or (j >= N):
            raise ValueError('Element refers to invalid node index.')
        xi = node_coords[i]
        xj = node_coords[j]
        E = float(elem['E'])
        nu = float(elem['nu'])
        A = float(elem['A'])
        Iy = float(elem['I_y'])
        Iz = float(elem['I_z'])
        J = float(elem['J'])
        local_z = elem.get('local_z', None)
        (R, L) = build_rotation(xi, xj, local_z)
        if L <= 0.0:
            raise ValueError('Element length must be positive.')
        G = E / (2.0 * (1.0 + nu))
        a = E * A / L
        t = G * J / L
        Bz = 12.0 * E * Iz / L ** 3
        Cz = 6.0 * E * Iz / L ** 2
        Dz = 4.0 * E * Iz / L
        Ez_ = 2.0 * E * Iz / L
        By = 12.0 * E * Iy / L ** 3
        Cy = 6.0 * E * Iy / L ** 2
        Dy = 4.0 * E * Iy / L
        Ey_ = 2.0 * E * Iy / L
        k_local = np.zeros((12, 12), dtype=float)
        k_local[0, 0] = a
        k_local[0, 6] = -a
        k_local[6, 0] = -a
        k_local[6, 6] = a
        k_local[3, 3] = t
        k_local[3, 9] = -t
        k_local[9, 3] = -t
        k_local[9, 9] = t
        k_local[1, 1] = Bz
        k_local[1, 5] = Cz
        k_local[1, 7] = -Bz
        k_local[1, 11] = Cz
        k_local[5, 1] = Cz
        k_local[5, 5] = Dz
        k_local[5, 7] = -Cz
        k_local[5, 11] = Ez_
        k_local[7, 1] = -Bz
        k_local[7, 5] = -Cz
        k_local[7, 7] = Bz
        k_local[7, 11] = -Cz
        k_local[11, 1] = Cz
        k_local[11, 5] = Ez_
        k_local[11, 7] = -Cz
        k_local[11, 11] = Dz
        k_local[2, 2] = By
        k_local[2, 4] = Cy
        k_local[2, 8] = -By
        k_local[2, 10] = Cy
        k_local[4, 2] = Cy
        k_local[4, 4] = Dy
        k_local[4, 8] = -Cy
        k_local[4, 10] = Ey_
        k_local[8, 2] = -By
        k_local[8, 4] = -Cy
        k_local[8, 8] = By
        k_local[8, 10] = -Cy
        k_local[10, 2] = Cy
        k_local[10, 4] = Ey_
        k_local[10, 8] = -Cy
        k_local[10, 10] = Dy
        T = np.zeros((12, 12), dtype=float)
        for b in range(4):
            T[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] = R
        k_global = T.T @ k_local @ T
        dofs = [6 * i + 0, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j + 0, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        idx = np.ix_(dofs, dofs)
        K[idx] += k_global
    fixed = np.zeros(ndof, dtype=bool)
    if boundary_conditions is not None:
        for (n, bc) in boundary_conditions.items():
            if n < 0 or n >= N:
                raise ValueError('boundary_conditions contains an invalid node index.')
            bc_arr = np.asarray(bc, dtype=int).reshape(-1)
            if bc_arr.size != 6:
                raise ValueError('Each boundary condition entry must have 6 components.')
            base = 6 * n
            for k in range(6):
                if bc_arr[k] != 0:
                    fixed[base + k] = True
    free = np.where(~fixed)[0]
    u = np.zeros(ndof, dtype=float)
    if free.size > 0:
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        cond = np.linalg.cond(K_ff)
        if not np.isfinite(cond) or cond >= 1e+16:
            raise ValueError('Ill-conditioned free–free stiffness submatrix (cond >= 1e16).')
        u_f = np.linalg.solve(K_ff, P_f)
        u[free] = u_f
    r = K @ u - P
    r[~fixed] = 0.0
    return (u, r)