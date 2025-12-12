def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    """
    Compute the local internal nodal force/moment vector (end loading) for a 3D Euler-Bernoulli beam element.
    applies the local stiffness matrix, and returns the corresponding internal end forces
    in the local coordinate system.
    Parameters
    ----------
    ele_info : dict
        Dictionary containing the element's material and geometric properties:
            'E' : float
                Young's modulus (Pa).
            'nu' : float
                Poisson's ratio (unitless).
            'A' : float
                Cross-sectional area (m²).
            'I_y', 'I_z' : float
                Second moments of area about the local y- and z-axes (m⁴).
            'J' : float
                Torsional constant (m⁴).
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    xi, yi, zi : float
        Global coordinates of the element's start node.
    xj, yj, zj : float
        Global coordinates of the element's end node.
    u_dofs_global : array-like of shape (12,)
        Element displacement vector in global coordinates:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2],
        where:
    Returns
    -------
    load_dofs_local : ndarray of shape (12,)
        Internal element end forces in local coordinates, ordered consistently with `u_dofs_global`.
        Positive forces/moments follow the local right-handed coordinate system conventions.
    Raises
    ------
    ValueError
        If the beam length is zero or if `local_z` is invalid.
    Notes
    -----
    elastic response to the provided displacement state, not externally applied loads.
    + local x-axis → element axis from node i to node j
    + local y-axis → perpendicular to both local z (provided) and x
    + local z-axis → defined by the provided (optional) or default orientation vector.
    [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i, Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j],
    consistent with the element’s local displacement ordering
    [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2].
    convention: tension and positive bending act in the positive local directions.
    """
    import numpy as np
    u = np.asarray(u_dofs_global, dtype=float).reshape(12)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    if L <= 0.0:
        raise ValueError('Beam length is zero.')
    ex = np.array([dx, dy, dz], dtype=float) / L
    local_z = ele_info.get('local_z', None)
    if local_z is None:
        ez_global = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(ex, ez_global)) > 0.999999:
            ez_global = np.array([0.0, 1.0, 0.0], dtype=float)
        local_z_vec = ez_global
    else:
        local_z_vec = np.asarray(local_z, dtype=float)
        if local_z_vec.shape != (3,):
            raise ValueError('local_z must be a 3-element vector.')
        if np.linalg.norm(local_z_vec) == 0.0:
            raise ValueError('local_z must be non-zero.')
    cross_v = np.cross(local_z_vec, ex)
    norm_cross = np.linalg.norm(cross_v)
    if norm_cross < 1e-12:
        raise ValueError('Provided local_z is parallel to the beam axis or invalid.')
    ey = cross_v / norm_cross
    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)
    R = np.column_stack((ex, ey, ez))
    RT = R.T
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = RT
    T[3:6, 3:6] = RT
    T[6:9, 6:9] = RT
    T[9:12, 9:12] = RT
    u_local = T.dot(u)
    E = float(ele_info.get('E', 0.0))
    nu = float(ele_info.get('nu', 0.0))
    A = float(ele_info.get('A', 0.0))
    I_y = float(ele_info.get('I_y', 0.0))
    I_z = float(ele_info.get('I_z', 0.0))
    J = float(ele_info.get('J', 0.0))
    if 1.0 + nu == 0.0:
        raise ValueError("Invalid Poisson's ratio leading to division by zero for G.")
    G = E / (2.0 * (1.0 + nu))
    K = np.zeros((12, 12), dtype=float)
    k_axial = E * A / L
    K[0, 0] += k_axial
    K[0, 6] += -k_axial
    K[6, 0] += -k_axial
    K[6, 6] += k_axial
    k_tors = G * J / L
    K[3, 3] += k_tors
    K[3, 9] += -k_tors
    K[9, 3] += -k_tors
    K[9, 9] += k_tors
    EI_z = E * I_z
    coeff_z = EI_z / L ** 3
    k_bz = coeff_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
    idx_bz = [1, 5, 7, 11]
    for a in range(4):
        for b in range(4):
            K[idx_bz[a], idx_bz[b]] += k_bz[a, b]
    EI_y = E * I_y
    coeff_y = EI_y / L ** 3
    k_by = coeff_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L * L, 6.0 * L, 2.0 * L * L], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L * L, 6.0 * L, 4.0 * L * L]], dtype=float)
    idx_by = [2, 4, 8, 10]
    for a in range(4):
        for b in range(4):
            K[idx_by[a], idx_by[b]] += k_by[a, b]
    load_local = K.dot(u_local)
    return np.asarray(load_local).reshape(12)