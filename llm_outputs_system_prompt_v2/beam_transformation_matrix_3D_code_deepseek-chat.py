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
    start_node = np.array([x1, y1, z1])
    end_node = np.array([x2, y2, z2])
    beam_vector = end_node - start_node
    beam_length = np.linalg.norm(beam_vector)
    if beam_length == 0:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    x_local = beam_vector / beam_length
    if reference_vector is not None:
        if reference_vector.shape != (3,):
            raise ValueError("The `reference_vector` doesn't have shape (3,).")
        ref_norm = np.linalg.norm(reference_vector)
        if abs(ref_norm - 1.0) > 1e-12:
            raise ValueError('`reference_vector` is not a unit vector.')
        y_local = np.cross(reference_vector, x_local)
        y_norm = np.linalg.norm(y_local)
        if y_norm < 1e-12:
            raise ValueError('`reference_vector` is parallel to the beam axis.')
        y_local = y_local / y_norm
        z_local = np.cross(x_local, y_local)
    else:
        global_z = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(np.cross(x_local, global_z)) < 1e-12:
            global_y = np.array([0.0, 1.0, 0.0])
            y_local = np.cross(global_y, x_local)
            y_norm = np.linalg.norm(y_local)
            if y_norm < 1e-12:
                y_local = np.cross(global_z, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
        else:
            y_local = np.cross(global_z, x_local)
            y_local = y_local / np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
    R = np.column_stack((x_local, y_local, z_local))
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return Gamma