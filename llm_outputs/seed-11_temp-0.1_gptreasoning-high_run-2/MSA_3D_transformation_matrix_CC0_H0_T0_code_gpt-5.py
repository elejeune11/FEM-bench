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
    v = np.array([x2 - x1, y2 - y1, z2 - z1], dtype=float)
    L = np.linalg.norm(v)
    if L == 0 or not np.isfinite(L):
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    x_hat = v / L
    atol_unit = 1e-08
    atol_align = 1e-08
    eps = 1e-12
    if reference_vector is None:
        gZ = np.array([0.0, 0.0, 1.0])
        gY = np.array([0.0, 1.0, 0.0])
        gX = np.array([1.0, 0.0, 0.0])
        if np.isclose(abs(np.dot(x_hat, gZ)), 1.0, atol=atol_align):
            r = gY.copy()
        else:
            r = gZ.copy()
        if np.linalg.norm(np.cross(r, x_hat)) <= eps:
            r = gY if not np.isclose(abs(np.dot(x_hat, gY)), 1.0, atol=atol_align) else gX
            if np.linalg.norm(np.cross(r, x_hat)) <= eps:
                raise ValueError('Failed to determine a valid default reference vector.')
    else:
        r_arr = np.asarray(reference_vector, dtype=float)
        if r_arr.shape != (3,):
            raise ValueError('reference_vector must have shape (3,).')
        rnorm = np.linalg.norm(r_arr)
        if not np.isclose(rnorm, 1.0, atol=atol_unit):
            raise ValueError('reference_vector must be a unit vector.')
        if np.isclose(abs(np.dot(r_arr, x_hat)), 1.0, atol=atol_align):
            raise ValueError('reference_vector cannot be parallel to the beam axis.')
        r = r_arr / rnorm
    y_temp = np.cross(r, x_hat)
    y_norm = np.linalg.norm(y_temp)
    if y_norm <= eps:
        raise ValueError('reference_vector leads to undefined local y-axis (parallel to beam axis).')
    y_hat = y_temp / y_norm
    z_temp = np.cross(x_hat, y_hat)
    z_norm = np.linalg.norm(z_temp)
    if z_norm <= eps:
        raise ValueError('Failed to compute a valid local z-axis.')
    z_hat = z_temp / z_norm
    R = np.vstack((x_hat, y_hat, z_hat))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in range(4):
        Gamma[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
    return Gamma