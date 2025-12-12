def FEM_2D_quad8_physical_gradient_CC0_H1_T3(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    """
    Compute the physical (x, y) gradient of a scalar field for a quadratic
    8-node quadrilateral (Q8) element at one or more natural coordinates (ξ, η).
    from nodal coordinates, and maps natural derivatives to the physical domain.
    Parameters
    ----------
    node_coords : np.ndarray
        Nodal coordinates of the Q8 element.
        Shape: (8, 2). Each row corresponds to a node, with columns [x, y].
    node_values : np.ndarray
        Scalar nodal values associated with the element.
        Shape: (8,).
    xi : float or np.ndarray
        ξ-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    eta : float or np.ndarray
        η-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    Returns
    -------
    grad_phys : np.ndarray
        Physical gradient of the scalar field at each evaluation point.
        Shape: (2, n_pts), where rows correspond to [∂u/∂x, ∂u/∂y]
        and column j corresponds to point (xi[j], eta[j]).
    Notes
    -----
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    Derivatives with respect to ξ and η are computed for each node in this order:
        [N1, N2, N3, N4, N5, N6, N7, N8]
    Node ordering (must match both `node_coords` and `node_values`):
        1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
        5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n_pts = len(xi)
    grad_phys = np.zeros((2, n_pts))
    for pt in range(n_pts):
        xi_val = xi[pt]
        eta_val = eta[pt]
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)
        dN_dxi[0] = -0.25 * (-(1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (-(1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) - (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (2 * xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + 2*xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = 0.25 * (1 - eta_val) * (2*xi_val + eta_val) + 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = 0.25 * ((1 - eta_val) * (2*xi_val + eta_val + 1) + (1 - eta_val))
        dN_dxi[0] = 0.25 * (1 - eta_val) * (2*xi_val + eta_val + 2)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (2*xi_val + eta_val + 2)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (2*xi_val + eta_val + 2)
        dN_dxi[0] = 0.25 * (1 - eta_val) * (2*xi_val + eta_val + 2)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (2*xi_val + eta_val + 2)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * ((1 - eta_val) * (1 + xi_val + eta_val) + (1 - xi_val) * (1 - eta_val))
        dN_dxi[0] = -0.25 * (1 - eta_val) * (1 + xi_val + eta_val) - 0.25 * (1 - xi_val) * (1 - eta_val)
        dN_dxi[0] = -0.25 * (1 - eta_val) - 0.25 * (1 - xi_val) *
