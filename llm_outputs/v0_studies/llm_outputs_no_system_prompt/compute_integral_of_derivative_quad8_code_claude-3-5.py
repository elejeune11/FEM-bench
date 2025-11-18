def compute_integral_of_derivative_quad8(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    """
    ∫_Ω (∇u) dΩ for a single quadratic quadrilateral element (scalar field only).
    Summary
    -------
    Computes the area integral of the physical gradient of a scalar field u
    over a Q8 8-node quadrilateral using Gauss–Legendre quadrature
    on the reference square Q = [-1,1]×[-1,1], mapped isoparametrically to Ω.
    Dependencies (only)
    -------------------
      used to assemble J(ξ,η) = node_coords^T @ dN and det(J).
      gives [∂u/∂x; ∂u/∂y] at quadrature points.
      Gauss-Legendre points/weights for num_pts ∈ {1,4,9}.
    Parameters
    ----------
    node_coords : (8,2) float
        Physical coordinates in Q8 node order:
          1: (-1,-1), 2: ( 1,-1), 3: ( 1, 1), 4: (-1, 1),
          5: ( 0,-1), 6: ( 1, 0), 7: ( 0, 1), 8: (-1, 0)
    node_values : (8,) or (8,1) float
        Scalar nodal values for u. (If (8,1), it is squeezed.)
    num_gauss_pts : {1,4,9}
        Total quadrature points (1×1, 2×2, or 3×3).
    Returns
    -------
    integral : (2,) float
        [∫_Ω ∂u/∂x dΩ, ∫_Ω ∂u/∂y dΩ].
    """
    (points, weights) = quad_quadrature_2D(num_gauss_pts)
    (_, dN) = quad8_shape_functions_and_derivatives(points)
    NC_T = node_coords.T
    det_J = np.zeros(num_gauss_pts)
    for p in range(num_gauss_pts):
        J = NC_T @ dN[p]
        det_J[p] = np.abs(np.linalg.det(J))
    grad_phys = compute_physical_gradient_quad8(node_coords, node_values, points[:, 0], points[:, 1])
    integral = np.zeros(2)
    for i in range(num_gauss_pts):
        integral += weights[i] * det_J[i] * grad_phys[:, i]
    return integral