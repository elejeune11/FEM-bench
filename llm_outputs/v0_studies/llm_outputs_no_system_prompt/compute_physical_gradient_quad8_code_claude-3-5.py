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
    xi_pts = np.atleast_1d(xi)
    eta_pts = np.atleast_1d(eta)
    n_pts = len(xi_pts)
    xi_eta = np.column_stack((xi_pts, eta_pts))
    (_, dN) = quad8_shape_functions_and_derivatives(xi_eta)
    grad_phys = np.zeros((2, n_pts))
    for i in range(n_pts):
        J = np.zeros((2, 2))
        J[0, 0] = np.sum(dN[i, :, 0] * node_coords[:, 0])
        J[0, 1] = np.sum(dN[i, :, 0] * node_coords[:, 1])
        J[1, 0] = np.sum(dN[i, :, 1] * node_coords[:, 0])
        J[1, 1] = np.sum(dN[i, :, 1] * node_coords[:, 1])
        grad_nat = np.array([np.sum(dN[i, :, 0] * node_values), np.sum(dN[i, :, 1] * node_values)])
        grad_phys[:, i] = np.linalg.solve(J.T, grad_nat)
    return grad_phys