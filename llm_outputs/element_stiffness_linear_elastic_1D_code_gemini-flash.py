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
    (points, weights) = gauss_quadrature_1D(n_gauss)
    dN_dxi = shape_function_derivatives_1D_linear()
    jacobian = compute_jacobian_1D(dN_dxi, x_elem)
    k_elem = np.zeros((2, 2))
    for i in range(n_gauss):
        B = dN_dxi / jacobian
        k_elem += weights[i] * B[:, None] * B[None, :] * E * A * jacobian
    return k_elem