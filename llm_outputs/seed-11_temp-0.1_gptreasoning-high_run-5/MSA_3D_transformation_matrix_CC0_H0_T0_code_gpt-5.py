def MSA_3D_transformation_matrix_CC0_H0_T0(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
    """
    Compute the 12x12 transformation matrix Gamma for a 3D beam element.
    This transformation relates the element's local coordinate system to the global system:
        K_global = Gamma.T @ K_local @ Gamma
    where K_global is the global stiffness matrix and K_local is the local stiffness matrix.
    Parameters:
        x1, y1, z1 (float): Coordinates of the beam's start node in global space.
        x2, y2, z2 (float): Coordinates of the beam's end node in global space.
        reference_vector (np.ndarray of shape (3,), optional): A unit vector in global coordinates used to define
            the orientation of the local y-axis. The local y-axis is computed as the cross product
            of the reference vector and the local x-axis (beam axis). The local z-axis is then
            computed as the cross product of the local x-axis and the local y-axes.
            If not provided:
    Returns:
        Gamma (np.ndarray): A 12x12 local-to-global transformation matrix used to transform
            stiffness matrices, displacements, and forces. It is composed of four repeated
            3x3 direction cosine submatrices along the diagonal.
    Raises:
        ValueError: If `reference_vector` is not a unit vector.
        ValueError: If `reference_vector` is parallel to the beam axis.
        ValueError: If the `reference_vector` doesn't have shape (3,).
        ValueError: If the beam has zero length (start and end nodes coincide).
    Notes:
        All vectors must be specified in a right-handed global Cartesian coordinate system.
    """
    tol = 1e-12
    p1 = np.array([x1, y1, z1], dtype=float)
    p2 = np.array([x2, y2, z2], dtype=float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if not np.isfinite(L) or L <= tol:
        raise ValueError('The beam has zero length or invalid coordinates.')
    x_axis = v / L
    if reference_vector is None:
        z_global = np.array([0.0, 0.0, 1.0])
        y_global = np.array([0.0, 1.0, 0.0])
        if np.isclose(abs(np.dot(x_axis, z_global)), 1.0, atol=1e-08):
            r = y_global
        else:
            r = z_global
    else:
        r = np.asarray(reference_vector, dtype=float)
        if r.shape != (3,):
            raise ValueError('reference_vector must have shape (3,).')
        if not np.all(np.isfinite(r)):
            raise ValueError('reference_vector contains non-finite values.')
        nr = np.linalg.norm(r)
        if not np.isclose(nr, 1.0, atol=1e-08):
            raise ValueError('reference_vector is not a unit vector.')
        if np.isclose(abs(np.dot(r, x_axis)), 1.0, atol=1e-08):
            raise ValueError('reference_vector is parallel to the beam axis.')
    y_temp = np.cross(r, x_axis)
    ny = np.linalg.norm(y_temp)
    if ny <= tol:
        raise ValueError('reference_vector is parallel or nearly parallel to the beam axis.')
    y_axis = y_temp / ny
    z_axis = np.cross(x_axis, y_axis)
    nz = np.linalg.norm(z_axis)
    if nz <= tol:
        raise ValueError('Failed to construct a valid orthonormal local coordinate system.')
    z_axis = z_axis / nz
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack((x_axis, y_axis, z_axis))
    Gamma = np.kron(np.eye(4), R)
    return Gamma