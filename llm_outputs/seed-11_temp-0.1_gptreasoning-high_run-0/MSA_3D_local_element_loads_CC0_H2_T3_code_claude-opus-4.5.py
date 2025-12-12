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
    consistent with the element's local displacement ordering
    [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2].
    convention: tension and positive bending act in the positive local directions.
    """
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    local_z = ele_info.get('local_z', None)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if L == 0:
        raise ValueError('Beam length is zero.')
    local_x = np.array([dx, dy, dz]) / L
    if local_z is None:
        global_z = np.array([0.0, 0.0, 1.0])
        global_y = np.array([0.0, 1.0, 0.0])
        if np.abs(np.abs(np.dot(local_x, global_z)) - 1.0) < 1e-10:
            local_z = global_y
        else:
            local_z = global_z
    else:
        local_z = np.array(local_z, dtype=float)
        local_z = local_z / np.linalg.norm(local_z)
    if np.abs(np.abs(np.dot(local_x, local_z)) - 1.0) < 1e-10:
        raise ValueError('local_z is parallel to the beam axis.')
    local_y = np.cross(local_z, local_x)
    local_y = local_y / np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)
    R3 = np.array([local_x, local_y, local_z])
    T = np.zeros((12, 12))
    T[0:3, 0:3] = R3
    T[3:6, 3:6] = R3
    T[6:9, 6:9] = R3
    T[9:12, 9:12] = R3
    u_global = np.array(u_dofs_global, dtype=float)
    u_local = T @ u_global
    G = E / (2.0 * (1.0 + nu))
    K_local = np.zeros((12, 12))
    k_axial = E * A / L
    K_local[0, 0] = k_axial
    K_local[0, 6] = -k_axial
    K_local[6, 0] = -k_axial
    K_local[6, 6] = k_axial
    k_torsion = G * J / L
    K_local[3, 3] = k_torsion
    K_local[3, 9] = -k_torsion
    K_local[9, 3] = -k_torsion
    K_local[9, 9] = k_torsion
    k1 = 12.0 * E * I_y / L ** 3
    k2 = 6.0 * E * I_y / L ** 2
    k3 = 4.0 * E * I_y / L
    k4 = 2.0 * E * I_y / L
    K_local[2, 2] = k1
    K_local[2, 4] = -k2
    K_local[2, 8] = -k1
    K_local[2, 10] = -k2
    K_local[4, 2] = -k2
    K_local[4, 4] = k3
    K_local[4, 8] = k2
    K_local[4, 10] = k4
    K_local[8, 2] = -k1
    K_local[8, 4] = k2
    K_local[8, 8] = k1
    K_local[8, 10] = k2
    K_local[10, 2] = -k2
    K_local[10, 4] = k4
    K_local[10, 8] = k2
    K_local[10, 10] = k3
    k1 = 12.0 * E * I_z / L ** 3
    k2 = 6.0 * E * I_z / L ** 2
    k3 = 4.0 * E * I_z / L
    k4 = 2.0 * E * I_z / L
    K_local[1, 1] = k1
    K_local[1, 5] = k2
    K_local[1, 7] = -k1
    K_local[1, 11] = k2
    K_local[5, 1] = k2
    K_local[5, 5] = k3
    K_local[5, 7] = -k2
    K_local[5, 11] = k4
    K_local[7, 1] = -k1
    K_local[7, 5] = -k2
    K_local[7, 7] = k1
    K_local[7, 11] = -k2
    K_local[11, 1] = k2
    K_local[11, 5] = k4
    K_local[11, 7] = -k2
    K_local[11, 11] = k3
    load_dofs_local = K_local @ u_local
    return load_dofs_local