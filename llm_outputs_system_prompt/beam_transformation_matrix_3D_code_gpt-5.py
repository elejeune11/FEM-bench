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
    tol = 1e-12
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if np.isclose(L, 0.0, atol=tol):
        raise ValueError('Beam has zero length.')
    ex = np.array([dx, dy, dz], dtype=float) / L
    if reference_vector is None:
        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        if np.isclose(abs(np.dot(ex, z_axis)), 1.0, atol=tol):
            ref = y_axis
        else:
            ref = z_axis
    else:
        ref_arr = np.asarray(reference_vector, dtype=float)
        if ref_arr.shape != (3,):
            raise ValueError('reference_vector must have shape (3,).')
        nref = float(np.linalg.norm(ref_arr))
        if not np.isclose(nref, 1.0, atol=tol):
            raise ValueError('reference_vector must be a unit vector.')
        ref = ref_arr
    cross_ref_ex = np.cross(ref, ex)
    norm_cross = float(np.linalg.norm(cross_ref_ex))
    if norm_cross < tol:
        raise ValueError('reference_vector is parallel or nearly parallel to the beam axis.')
    ey = cross_ref_ex / norm_cross
    ez = np.cross(ex, ey)
    nez = float(np.linalg.norm(ez))
    if nez < tol:
        raise ValueError('Failed to construct a valid local coordinate system.')
    ez = ez / nez
    R = np.vstack((ex, ey, ez))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in range(4):
        Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R
    return Gamma