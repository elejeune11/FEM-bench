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
    u = np.asarray(u_dofs_global, dtype=float).reshape(-1)
    if u.size != 12:
        raise ValueError('u_dofs_global must be an array-like of length 12.')
    dx = float(xj) - float(xi)
    dy = float(yj) - float(yi)
    dz = float(zj) - float(zi)
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError('Element length must be positive and finite.')
    x_hat = np.array([dx, dy, dz], dtype=float) / L
    tol_parallel = 1e-08
    z_provided = None
    if isinstance(ele_info, dict) and 'local_z' in ele_info:
        z_provided = ele_info.get('local_z', None)
    if z_provided is None:
        z0 = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(z0, x_hat))) > 1.0 - tol_parallel:
            z0 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        z0 = np.asarray(z_provided, dtype=float).reshape(-1)
        if z0.size != 3:
            raise ValueError('local_z must be a 3-component vector.')
        nrm = float(np.linalg.norm(z0))
        if not np.isfinite(nrm) or nrm <= 0.0:
            raise ValueError('local_z vector must be nonzero and finite.')
        z0 = z0 / nrm
        if abs(float(np.dot(z0, x_hat))) > 1.0 - tol_parallel:
            raise ValueError('local_z must not be parallel to the beam axis.')
    y_hat = np.cross(z0, x_hat)
    nrm_y = float(np.linalg.norm(y_hat))
    if nrm_y <= 1e-12:
        raise ValueError('Invalid local_z: it is (near) parallel to the beam axis.')
    y_hat = y_hat / nrm_y
    z_hat = np.cross(x_hat, y_hat)
    z_hat = z_hat / float(np.linalg.norm(z_hat))
    R = np.vstack((x_hat, y_hat, z_hat))
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    T[6:9, 6:9] = R
    T[9:12, 9:12] = R
    u_local = T @ u
    E = float(ele_info['E'])
    nu = float(ele_info['nu'])
    A = float(ele_info['A'])
    Iy = float(ele_info['I_y'])
    Iz = float(ele_info['I_z'])
    J = float(ele_info['J'])
    G = E / (2.0 * (1.0 + nu))
    K = np.zeros((12, 12), dtype=float)
    k_ax = E * A / L
    K[0, 0] += k_ax
    K[0, 6] += -k_ax
    K[6, 0] += -k_ax
    K[6, 6] += k_ax
    k_tor = G * J / L
    K[3, 3] += k_tor
    K[3, 9] += -k_tor
    K[9, 3] += -k_tor
    K[9, 9] += k_tor
    a = 12.0 * E * Iz / L ** 3
    b = 6.0 * E * Iz / L ** 2
    c = 4.0 * E * Iz / L
    d = 2.0 * E * Iz / L
    K[1, 1] += a
    K[1, 5] += b
    K[1, 7] += -a
    K[1, 11] += b
    K[5, 1] += b
    K[5, 5] += c
    K[5, 7] += -b
    K[5, 11] += d
    K[7, 1] += -a
    K[7, 5] += -b
    K[7, 7] += a
    K[7, 11] += -b
    K[11, 1] += b
    K[11, 5] += d
    K[11, 7] += -b
    K[11, 11] += c
    a2 = 12.0 * E * Iy / L ** 3
    b2 = 6.0 * E * Iy / L ** 2
    c2 = 4.0 * E * Iy / L
    d2 = 2.0 * E * Iy / L
    K[2, 2] += a2
    K[2, 4] += -b2
    K[2, 8] += -a2
    K[2, 10] += -b2
    K[4, 2] += -b2
    K[4, 4] += c2
    K[4, 8] += b2
    K[4, 10] += d2
    K[8, 2] += -a2
    K[8, 4] += b2
    K[8, 8] += a2
    K[8, 10] += b2
    K[10, 2] += -b2
    K[10, 4] += d2
    K[10, 8] += b2
    K[10, 10] += c2
    load_dofs_local = K @ u_local
    return load_dofs_local