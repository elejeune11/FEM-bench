def MSA_3D_local_element_loads_CC0_H2_T1(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
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
      to the provided displacement state, not externally applied loads.
    Support Functions Used
    ----------------------
        Computes the 12x12 transformation matrix (Gamma) relating local and global coordinate systems
        for a 3D beam element. Ensures orthonormal local axes and validates the reference vector.
        Returns the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with the
        local x-axis, capturing axial, bending, and torsional stiffness.
    """
    u_global = np.asarray(u_dofs_global, dtype=float).ravel()
    if u_global.size != 12:
        raise ValueError('u_dofs_global must be of length 12.')
    u_global = u_global.reshape(12)
    try:
        E = float(ele_info['E'])
        nu = float(ele_info['nu'])
        A = float(ele_info['A'])
        J = float(ele_info['J'])
    except Exception as e:
        raise ValueError("Missing required element property 'E', 'nu', 'A', or 'J'.") from e
    if 'I_y' in ele_info:
        Iy = float(ele_info['I_y'])
    elif 'Iy' in ele_info:
        Iy = float(ele_info['Iy'])
    else:
        raise ValueError("Missing required element property 'I_y' or 'Iy'.")
    if 'I_z' in ele_info:
        Iz = float(ele_info['I_z'])
    elif 'Iz' in ele_info:
        Iz = float(ele_info['Iz'])
    else:
        raise ValueError("Missing required element property 'I_z' or 'Iz'.")
    local_z = ele_info.get('local_z', None)
    if local_z is not None:
        local_z = np.asarray(local_z, dtype=float)
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    if np.isclose(L, 0.0):
        raise ValueError('Beam length is zero.')
    k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    u_local = Gamma @ u_global
    load_dofs_local = k_local @ u_local
    return load_dofs_local