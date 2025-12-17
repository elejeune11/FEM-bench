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
    import numpy as np
    tol_zero = 1e-12
    tol_unit = 1e-08
    tol_parallel = 1e-12
    v = np.array([x2 - x1, y2 - y1, z2 - z1], dtype=float)
    L = np.linalg.norm(v)
    if not np.isfinite(L) or L < tol_zero:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    ex = v / L
    if reference_vector is None:
        gz = np.array([0.0, 0.0, 1.0], dtype=float)
        gy = np.array([0.0, 1.0, 0.0], dtype=float)
        if np.linalg.norm(np.cross(ex, gz)) < tol_parallel:
            ref = gy
        else:
            ref = gz
    else:
        ref_arr = np.asarray(reference_vector, dtype=float)
        if ref_arr.ndim != 1 or ref_arr.shape[0] != 3:
            raise ValueError("The `reference_vector` doesn't have shape (3,).")
        nref = np.linalg.norm(ref_arr)
        if not np.isfinite(nref) or abs(nref - 1.0) > tol_unit:
            raise ValueError('`reference_vector` is not a unit vector.')
        ref = ref_arr
        if np.linalg.norm(np.cross(ref, ex)) < tol_parallel:
            raise ValueError('`reference_vector` is parallel to the beam axis.')
    y_temp = np.cross(ref, ex)
    ny = np.linalg.norm(y_temp)
    if ny < tol_parallel:
        raise ValueError('Failed to construct a valid local y-axis; reference is parallel to the beam axis.')
    ey = y_temp / ny
    ez = np.cross(ex, ey)
    R = np.vstack((ex, ey, ez))
    Gamma = np.kron(np.eye(4), R)
    return Gamma