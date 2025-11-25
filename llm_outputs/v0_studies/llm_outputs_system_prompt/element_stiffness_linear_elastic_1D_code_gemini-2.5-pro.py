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
    k_elem = np.zeros((2, 2))
    (_, gauss_weights) = gauss_quadrature_1D(n_gauss)
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x_elem)
    if np.isclose(J, 0.0):
        return k_elem
    B = 1.0 / J * dN_dxi
    B_mat = B.reshape(1, 2)
    integrand = B_mat.T @ B_mat * E * A * np.abs(J)
    for w in gauss_weights:
        k_elem += integrand * w
    return k_elem