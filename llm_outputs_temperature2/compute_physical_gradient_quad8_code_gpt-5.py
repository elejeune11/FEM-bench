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
    nc = np.asarray(node_coords, dtype=float)
    if nc.shape != (8, 2):
        raise ValueError('node_coords must have shape (8, 2).')
    if not np.all(np.isfinite(nc)):
        raise ValueError('node_coords must contain finite values.')
    nv = np.asarray(node_values, dtype=float).reshape(-1)
    if nv.shape != (8,):
        raise ValueError('node_values must have shape (8,).')
    if not np.all(np.isfinite(nv)):
        raise ValueError('node_values must contain finite values.')
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=float))
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=float))
    if xi_arr.ndim != 1 or eta_arr.ndim != 1:
        raise ValueError('xi and eta must be scalars or 1D arrays.')
    if xi_arr.size != eta_arr.size:
        if xi_arr.size == 1:
            xi_arr = np.full_like(eta_arr, xi_arr.item(), dtype=float)
        elif eta_arr.size == 1:
            eta_arr = np.full_like(xi_arr, eta_arr.item(), dtype=float)
        else:
            raise ValueError('xi and eta must have the same length or be broadcastable (one can be scalar).')
    if not (np.all(np.isfinite(xi_arr)) and np.all(np.isfinite(eta_arr))):
        raise ValueError('xi and eta must contain finite values.')
    xi_eta = np.column_stack((xi_arr, eta_arr))
    (_, dN) = quad8_shape_functions_and_derivatives(xi_eta)
    dN_dxi = dN[:, :, 0]
    dN_deta = dN[:, :, 1]
    x = nc[:, 0]
    y = nc[:, 1]
    a = dN_dxi @ x
    b = dN_deta @ x
    c = dN_dxi @ y
    d = dN_deta @ y
    detJ = a * d - b * c
    if not np.all(np.isfinite(detJ)):
        raise ValueError('Non-finite Jacobian determinant encountered.')
    if np.any(np.abs(detJ) < 1e-14):
        raise ValueError('Singular or near-singular Jacobian encountered.')
    gxi = dN_dxi @ nv
    geta = dN_deta @ nv
    gx = (d * gxi - c * geta) / detJ
    gy = (-b * gxi + a * geta) / detJ
    return np.vstack((gx, gy))