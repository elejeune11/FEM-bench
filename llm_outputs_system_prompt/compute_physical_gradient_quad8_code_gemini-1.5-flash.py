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
    xi_eta = np.array([xi, eta]).T
    (N, dN_dxi) = quad8_shape_functions_and_derivatives(xi_eta)
    jacobian = dN_dxi @ node_coords
    natural_grad = dN_dxi.transpose((0, 2, 1)) @ node_values.reshape(-1, 1)
    jacobian_inv = np.linalg.inv(jacobian)
    grad_phys = jacobian_inv @ natural_grad
    return grad_phys.reshape(2, -1)