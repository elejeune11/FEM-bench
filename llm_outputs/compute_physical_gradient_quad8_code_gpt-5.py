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
        raise ValueError('node_coords must have shape (8,2).')
    if not np.all(np.isfinite(nc)):
        raise ValueError('node_coords must contain finite values.')
    nv = np.asarray(node_values, dtype=float)
    if nv.size != 8:
        raise ValueError('node_values must have size 8.')
    nv = nv.reshape(8)
    if not np.all(np.isfinite(nv)):
        raise ValueError('node_values must contain finite values.')
    xi_arr = np.asarray(xi, dtype=float)
    eta_arr = np.asarray(eta, dtype=float)
    if xi_arr.ndim == 0:
        xi_arr = xi_arr.reshape(1)
    if eta_arr.ndim == 0:
        eta_arr = eta_arr.reshape(1)
    try:
        (xi_b, eta_b) = np.broadcast_arrays(xi_arr, eta_arr)
    except ValueError as e:
        raise ValueError('xi and eta must be broadcastable to the same shape.') from e
    n_pts = xi_b.size
    pts = np.stack([xi_b.ravel(), eta_b.ravel()], axis=1)
    (_, dN) = quad8_shape_functions_and_derivatives(pts)
    grad_phys = np.empty((2, n_pts), dtype=float)
    coordsT = nc.T
    for p in range(n_pts):
        dNp = dN[p, :, :]
        J = coordsT @ dNp
        g_nat = nv @ dNp
        g_phys = np.linalg.solve(J.T, g_nat)
        grad_phys[:, p] = g_phys
    return grad_phys