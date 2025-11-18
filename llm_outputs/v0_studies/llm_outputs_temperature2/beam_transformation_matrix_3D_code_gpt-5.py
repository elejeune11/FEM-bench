def beam_transformation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
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
    p1 = np.array([x1, y1, z1], dtype=float)
    p2 = np.array([x2, y2, z2], dtype=float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L <= 1e-12:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    x_axis = v / L
    if reference_vector is None:
        ez = np.array([0.0, 0.0, 1.0])
        ey = np.array([0.0, 1.0, 0.0])
        if np.isclose(abs(np.dot(x_axis, ez)), 1.0, atol=1e-08):
            ref = ey
        else:
            ref = ez
    else:
        ref_arr = np.asarray(reference_vector, dtype=float)
        if ref_arr.shape != (3,):
            raise ValueError("The `reference_vector` doesn't have shape (3,).")
        nref = np.linalg.norm(ref_arr)
        if not np.isclose(nref, 1.0, atol=1e-08):
            raise ValueError('`reference_vector` is not a unit vector.')
        ref = ref_arr / nref
        if abs(np.dot(ref, x_axis)) >= 1.0 - 1e-08:
            raise ValueError('`reference_vector` is parallel to the beam axis.')
    y_temp = np.cross(ref, x_axis)
    ny = np.linalg.norm(y_temp)
    if ny <= 1e-12:
        raise ValueError('Failed to construct a valid local y-axis; reference may be parallel to beam axis.')
    y_axis = y_temp / ny
    z_axis = np.cross(x_axis, y_axis)
    nz = np.linalg.norm(z_axis)
    if nz <= 1e-12:
        raise ValueError('Failed to construct a valid local z-axis.')
    z_axis = z_axis / nz
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R = np.vstack((x_axis, y_axis, z_axis))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in (0, 3, 6, 9):
        Gamma[i:i + 3, i:i + 3] = R
    return Gamma