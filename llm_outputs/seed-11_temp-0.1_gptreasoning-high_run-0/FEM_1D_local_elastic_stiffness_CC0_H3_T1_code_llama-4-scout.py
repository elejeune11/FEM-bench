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
    dN_dxi = shape_function_derivatives_1D_linear()
    jac = compute_jacobian_1D(dN_dxi, x_elem)
    (gauss_points, gauss_weights) = gauss_quadrature_1D(n_gauss)
    K_elem = np.zeros((2, 2))
    for (i, (point, weight)) in enumerate(zip(gauss_points, gauss_weights)):
        B = dN_dxi / jac
        D = E * A
        K_elem += B[:, None] * B[None, :] * D * jac * weight
    return K_elem