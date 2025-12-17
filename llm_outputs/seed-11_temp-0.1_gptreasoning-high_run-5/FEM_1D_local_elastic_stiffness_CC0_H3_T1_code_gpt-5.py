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
    import numpy as np
    if n_gauss is None:
        n_gauss = 2
    x_elem = np.asarray(x_elem, dtype=float).reshape(-1)
    if x_elem.size != 2:
        raise ValueError('x_elem must be an array-like of length 2.')
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x_elem)
    if not np.isfinite(J) or J == 0.0:
        raise ValueError('Degenerate element: zero or invalid Jacobian.')
    dN_dx = dN_dxi / J
    J_abs = np.abs(J)
    _, weights = gauss_quadrature_1D(int(n_gauss))
    weight_sum = np.sum(weights)
    K = E * A * J_abs * weight_sum * np.outer(dN_dx, dN_dx)
    return K