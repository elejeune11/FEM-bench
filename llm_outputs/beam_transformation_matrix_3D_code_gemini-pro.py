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
    x = x2 - x1
    y = y2 - y1
    z = z2 - z1
    L = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if L == 0:
        raise ValueError('Beam has zero length.')
    lx = x / L
    ly = y / L
    lz = z / L
    if reference_vector is not None:
        if reference_vector.shape != (3,):
            raise ValueError('reference_vector must have shape (3,)')
        if not np.isclose(np.linalg.norm(reference_vector), 1.0):
            raise ValueError('reference_vector must be a unit vector.')
        if np.all(np.cross(reference_vector, np.array([lx, ly, lz])) == 0):
            raise ValueError('reference_vector cannot be parallel to the beam axis.')
        vy = np.cross(reference_vector, np.array([lx, ly, lz]))
        vy /= np.linalg.norm(vy)
    elif np.isclose(lx, 0.0) and np.isclose(ly, 0.0):
        vy = np.array([0.0, 1.0, 0.0])
    else:
        vy = np.cross(np.array([0, 0, 1]), np.array([lx, ly, lz]))
        vy /= np.linalg.norm(vy)
    vz = np.cross(np.array([lx, ly, lz]), vy)
    R = np.array([[lx, ly, lz], [vy[0], vy[1], vy[2]], [vz[0], vz[1], vz[2]]])
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[i * 3:i * 3 + 3, i * 3:i * 3 + 3] = R
    return Gamma