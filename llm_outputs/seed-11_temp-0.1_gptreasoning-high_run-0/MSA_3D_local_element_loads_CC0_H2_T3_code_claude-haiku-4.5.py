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
    u_dofs_global = np.asarray(u_dofs_global, dtype=float)
    E = ele_info['E']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    beam_vec = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
    L = np.linalg.norm(beam_vec)
    if L == 0:
        raise ValueError('Beam length is zero')
    local_x = beam_vec / L
    if 'local_z' in ele_info and ele_info['local_z'] is not None:
        local_z = np.asarray(ele_info['local_z'], dtype=float)
        local_z = local_z / np.linalg.norm(local_z)
    else:
        local_z = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(abs(np.dot(local_x, local_z)) - 1.0) < 1e-10:
            local_z = np.array([0.0, 1.0, 0.0], dtype=float)
    local_y = np.cross(local_z, local_x)
    local_y_norm = np.linalg.norm(local_y)
    if local_y_norm < 1e-10:
        raise ValueError('local_z is parallel to beam axis')
    local_y = local_y / local_y_norm
    local_z = np.cross(local_x, local_y)
    R = np.array([local_x, local_y, local_z], dtype=float)
    T = np.zeros((12, 12), dtype=float)
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        T[3 * i + 9:3 * i + 12, 3 * i + 9:3 * i + 12] = R
    u_dofs_local = T @ u_dofs_global
    EA_L = E * A / L
    EI_y_L3 = E * I_y / L ** 3
    EI_z_L3 = E * I_z / L ** 3
    GJ_L = E / (2 * (1 + ele_info.get('nu', 0.3))) * J / L
    k_local = np.zeros((12, 12), dtype=float)
    k_local[0, 0] = EA_L
    k_local[0, 6] = -EA_L
    k_local[6, 0] = -EA_L
    k_local[6, 6] = EA_L
    k_local[3, 3] = GJ_L
    k_local[3, 9] = -GJ_L
    k_local[9, 3] = -GJ_L
    k_local[9, 9] = GJ_L
    k_local[1, 1] = 12 * EI_z_L3
    k_local[1, 5] = 6 * EI_z_L3 * L
    k_local[1, 7] = -12 * EI_z_L3
    k_local[1, 11] = 6 * EI_z_L3 * L
    k_local[5, 1] = 6 * EI_z_L3 * L
    k_local[5, 5] = 4 * EI_z_L3 * L * L
    k_local[5, 7] = -6 * EI_z_L3 * L
    k_local[5, 11] = 2 * EI_z_L3 * L * L
    k_local[7, 1] = -12 * EI_z_L3
    k_local[7, 5] = -6 * EI_z_L3 * L
    k_local[7, 7] = 12 * EI_z_L3
    k_local[7, 11] = -6 * EI_z_L3 * L
    k_local[11, 1] = 6 * EI_z_L3 * L
    k_local[11, 5] = 2 * EI_z_L3 * L * L
    k_local[11, 7] = -6 * EI_z_L3 * L
    k_local[11, 11] = 4 * EI_z_L3 * L * L
    k_local[2, 2] = 12 * EI_y_L3
    k_local[2, 4] = -6 * EI_y_L3 * L
    k_local[2, 8] = -12 * EI_y_L3
    k_local[2, 10] = -6 * EI_y_L3 * L
    k_local[4, 2] = -6 * EI_y_L3 * L
    k_local[4, 4] = 4 * EI_y_L3 * L * L
    k_local[4, 8] = 6 * EI_y_L3 * L
    k_local[4, 10] = 2 * EI_y_L3 * L * L
    k_local[8, 2] = -12 * EI_y_L3
    k_local[8, 4] = 6 * EI_y_L3 * L
    k_local[8, 8] = 12 * EI_y_L3
    k_local[8, 10] = 6 * EI_y_L3 * L
    k_local[10, 2] = -6 * EI_y_L3 * L
    k_local[10, 4] = 2 * EI_y_L3 * L * L
    k_local[10, 8] = 6 * EI_y_L3 * L
    k_local[10, 10] = 4 * EI_y_L3 * L * L
    load_dofs_local = k_local @ u_dofs_local
    return load_dofs_local