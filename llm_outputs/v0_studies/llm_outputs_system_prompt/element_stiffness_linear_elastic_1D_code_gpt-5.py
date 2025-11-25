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
    x = np.asarray(x_elem, dtype=float).ravel()
    if x.size != 2:
        raise ValueError('x_elem must contain exactly two nodal coordinates.')
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x)
    absJ = float(abs(J))
    if absJ == 0.0:
        raise ValueError('Degenerate element with zero Jacobian.')
    (_, weights) = gauss_quadrature_1D(n_gauss)
    B = dN_dxi / J
    K = E * A * absJ * weights.sum() * np.outer(B, B)
    return K