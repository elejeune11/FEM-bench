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
    x_elem = np.asarray(x_elem, dtype=float).reshape(-1)
    if x_elem.size != 2:
        raise ValueError('x_elem must contain exactly two nodal coordinates [x1, x2].')
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x_elem)
    if J == 0.0:
        raise ValueError('Degenerate element with zero Jacobian (coincident nodes).')
    J_abs = abs(J)
    B = dN_dxi / J
    _, weights = gauss_quadrature_1D(n_gauss)
    K = np.zeros((2, 2), dtype=float)
    for w in weights:
        K += np.outer(B, B) * (E * A) * (J_abs * w)
    return K