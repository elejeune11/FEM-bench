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
    E = ele_info['E']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    local_z = ele_info.get('local_z', None)
    x_vec = np.array([xj - xi, yj - yi, zj - zi])
    length = np.linalg.norm(x_vec)
    if length == 0:
        raise ValueError('Beam length is zero.')
    x_unit = x_vec / length
    if local_z is None:
        if np.allclose(x_unit, [0, 0, 1]):
            local_z = np.array([0, 1, 0])
        else:
            local_z = np.array([0, 0, 1])
    local_z = np.array(local_z)
    local_z = local_z / np.linalg.norm(local_z)
    if np.allclose(local_z, x_unit) or np.allclose(local_z, -x_unit):
        raise ValueError('local_z is invalid.')
    y_unit = np.cross(local_z, x_unit)
    y_unit = y_unit / np.linalg.norm(y_unit)
    z_unit = np.cross(x_unit, y_unit)
    rotation_matrix = np.column_stack((x_unit, y_unit, z_unit))
    R = np.block([[rotation_matrix, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), rotation_matrix, np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), rotation_matrix, np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), rotation_matrix]])
    u_dofs_local = R @ u_dofs_global
    K_local = np.zeros((12, 12))
    K_local[0, 0] = E * A / length
    K_local[0, 6] = -E * A / length
    K_local[6, 0] = -E * A / length
    K_local[6, 6] = E * A / length
    K_local[3, 3] = E * J / length
    K_local[3, 9] = -E * J / length
    K_local[9, 3] = -E * J / length
    K_local[9, 9] = E * J / length
    K_local[1, 1] = 12 * E * I_z / length ** 3
    K_local[1, 5] = 6 * E * I_z / length ** 2
    K_local[1, 7] = -12 * E * I_z / length ** 3
    K_local[1, 11] = 6 * E * I_z / length ** 2
    K_local[5, 1] = 6 * E * I_z / length ** 2
    K_local[5, 5] = 4 * E * I_z / length
    K_local[5, 7] = -6 * E * I_z / length ** 2
    K_local[5, 11] = 2 * E * I_z / length
    K_local[7, 1] = -12 * E * I_z / length ** 3
    K_local[7, 5] = -6 * E * I_z / length ** 2
    K_local[7, 7] = 12 * E * I_z / length ** 3
    K_local[7, 11] = -6 * E * I_z / length ** 2
    K_local[11, 1] = 6 * E * I_z / length ** 2
    K_local[11, 5] = 2 * E * I_z / length
    K_local[11, 7] = -6 * E * I_z / length ** 2
    K_local[11, 11] = 4 * E * I_z / length
    K_local[2, 2] = 12 * E * I_y / length ** 3
    K_local[2, 4] = -6 * E * I_y / length ** 2
    K_local[2, 8] = -12 * E * I_y / length ** 3
    K_local[2, 10] = -6 * E * I_y / length ** 2
    K_local[4, 2] = -6 * E * I_y / length ** 2
    K_local[4, 4] = 4 * E * I_y / length
    K_local[4, 8] = 6 * E * I_y / length ** 2
    K_local[4, 10] = 2 * E * I_y / length
    K_local[8, 2] = -12 * E * I_y / length ** 3
    K_local[8, 4] = 6 * E * I_y / length ** 2
    K_local[8, 8] = 12 * E * I_y / length ** 3
    K_local[8, 10] = 6 * E * I_y / length ** 2
    K_local[10, 2] = -6 * E * I_y / length ** 2
    K_local[10, 4] = 2 * E * I_y / length
    K_local[10, 8] = 6 * E * I_y / length ** 2
    K_local[10, 10] = 4 * E * I_y / length
    load_dofs_local = K_local @ u_dofs_local
    return load_dofs_local