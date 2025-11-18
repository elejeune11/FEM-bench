def compute_physical_gradient_quad8(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    """
    Compute the physical (x,y) gradient of a scalar field for a quadratic
    8-node quadrilateral (Q8) at one or more natural-coordinate points (xi, eta).
    Steps:
      1) evaluate Q8 shape-function derivatives at (xi, eta),
      2) form the Jacobian from nodal coordinates,
      3) build the natural gradient from nodal values,
      4) map to physical coordinates using the Jacobian.
    Parameters
    ----------
    node_coords : (8,2)
        Physical coordinates of the Q8 nodes.
    node_values : (8,)
        Scalar nodal values.
    xi, eta : scalar or array-like (n_pts,)
        Natural coordinates of evaluation points.
    Assumptions / Conventions
    -------------------------
    Uses the Q8 shape functions exactly as in
        `quad8_shape_functions_and_derivatives` with natural domain [-1, 1]^2.
    Expected node ordering (must match both `node_coords` and the shape functions):
        1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
        5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    Passing nodes in a different order will produce incorrect results.
    Returns
    -------
    grad_phys : (2, n_pts)
        Rows are [∂u/∂x, ∂u/∂y] at each point.
        Column j corresponds to the j-th input point (xi[j], eta[j]).
    """
    if not isinstance(node_coords, np.ndarray):
        raise ValueError('node_coords must be a NumPy array.')
    if node_coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8,2).')
    if not np.all(np.isfinite(node_coords)):
        raise ValueError('node_coords must contain finite values.')
    X = node_coords.astype(float, copy=False)
    if not isinstance(node_values, np.ndarray):
        raise ValueError('node_values must be a NumPy array.')
    if node_values.shape not in [(8,), (8, 1)]:
        raise ValueError('node_values must have shape (8,) or (8,1).')
    if not np.all(np.isfinite(node_values)):
        raise ValueError('node_values must contain finite values.')
    u = node_values.astype(float, copy=False).reshape(8)
    xi_arr = np.asarray(xi, dtype=float)
    eta_arr = np.asarray(eta, dtype=float)
    if not np.all(np.isfinite(xi_arr)) or not np.all(np.isfinite(eta_arr)):
        raise ValueError('xi and eta must contain finite values.')
    (xi_b, eta_b) = np.broadcast_arrays(np.atleast_1d(xi_arr), np.atleast_1d(eta_arr))
    xi_flat = xi_b.ravel()
    eta_flat = eta_b.ravel()
    n = xi_flat.size
    xi_pts = np.column_stack((xi_flat, eta_flat))
    (_, dN) = quad8_shape_functions_and_derivatives(xi_pts)
    dN_dxi = dN[:, :, 0]
    dN_deta = dN[:, :, 1]
    du_dxi = dN_dxi @ u
    du_deta = dN_deta @ u
    x_i = X[:, 0]
    y_i = X[:, 1]
    J11 = dN_dxi @ x_i
    J12 = dN_deta @ x_i
    J21 = dN_dxi @ y_i
    J22 = dN_deta @ y_i
    detJ = J11 * J22 - J12 * J21
    if not np.all(np.isfinite(detJ)):
        raise ValueError('Jacobian determinant contains non-finite values.')
    if np.any(detJ == 0.0):
        raise np.linalg.LinAlgError('Singular Jacobian encountered at one or more points.')
    grad_x = (J22 * du_dxi - J21 * du_deta) / detJ
    grad_y = (-J12 * du_dxi + J11 * du_deta) / detJ
    grad = np.empty((2, n), dtype=float)
    grad[0, :] = grad_x
    grad[1, :] = grad_y
    return grad