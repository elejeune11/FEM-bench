def element_stiffness_linear_elastic_1D(x_elem: np.ndarray, E: float, A: float, n_gauss: int) -> np.ndarray:
    """
    Compute the element stiffness matrix for a 1D linear bar using the Galerkin method.
    Parameters:
        x_elem (np.ndarray): Nodal coordinates of the element [x1, x2]
        E (float): Young's modulus
        A (float): Cross-sectional area
        n_gauss (int): Number of Gauss integration points (default = 2)
    Returns:
        np.ndarray: 2x2 element stiffness matrix
    """
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x_elem)
    if np.isclose(J, 0.0):
        raise ValueError('Degenerate element: zero Jacobian.')
    J_abs = abs(J)
    B = dN_dxi / J_abs
    (_, weights) = gauss_quadrature_1D(n_gauss)
    K = np.zeros((2, 2), dtype=float)
    for w in weights:
        K += E * A * np.outer(B, B) * J_abs * w
    return K