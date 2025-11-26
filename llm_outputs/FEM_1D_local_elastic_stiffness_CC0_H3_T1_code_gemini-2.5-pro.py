def FEM_1D_local_elastic_stiffness_CC0_H3_T1(x_elem: np.ndarray, E: float, A: float, n_gauss: int) -> np.ndarray:
    """
    Compute the local stiffness matrix for a 1D linear elastic bar element 
    using the Galerkin finite element formulation.
    Parameters:
        x_elem (np.ndarray): Array of nodal coordinates for the element [x1, x2]
            (shape: [2,]).
        E (float): Young's modulus of the material.
        A (float): Cross-sectional area of the bar.
        n_gauss (int, optional): Number of Gauss integration points. 
            Defaults to 2, which is exact for linear elements.
    Returns:
        np.ndarray: 2×2 element stiffness matrix representing 
            the relation between nodal displacements and forces.
    """

    def gauss_quadrature_1D(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n == 1:
            points = np.array([0.0])
            weights = np.array([2.0])
        elif n == 2:
            sqrt_3_inv = 1.0 / np.sqrt(3)
            points = np.array([-sqrt_3_inv, sqrt_3_inv])
            weights = np.array([1.0, 1.0])
        elif n == 3:
            points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
            weights = np.array([5 / 9, 8 / 9, 5 / 9])
        else:
            raise ValueError('Only 1 to 3 Gauss points are supported.')
        return (points, weights)

    def shape_function_derivatives_1D_linear() -> np.ndarray:
        return np.array([-0.5, 0.5])

    def compute_jacobian_1D(dN_dxi: np.ndarray, x_elem: np.ndarray) -> float:
        return np.dot(dN_dxi, x_elem)
    k_local = np.zeros((2, 2))
    (_, gauss_weights) = gauss_quadrature_1D(n_gauss)
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x_elem)
    integrand = E * A / J * np.outer(dN_dxi, dN_dxi)
    for weight in gauss_weights:
        k_local += integrand * weight
    return k_local