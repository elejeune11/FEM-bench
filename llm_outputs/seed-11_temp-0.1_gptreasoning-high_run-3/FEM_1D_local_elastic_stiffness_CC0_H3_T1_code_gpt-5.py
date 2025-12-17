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
    import pytest
    x_elem = np.asarray(x_elem, dtype=float).reshape(-1)
    if x_elem.size != 2:
        raise ValueError('x_elem must be a 1D array of length 2: [x1, x2].')
    _, weights = gauss_quadrature_1D(int(n_gauss))
    dN_dxi = shape_function_derivatives_1D_linear()
    J = compute_jacobian_1D(dN_dxi, x_elem)
    J_abs = float(abs(J))
    if J_abs == 0.0 or not np.isfinite(J_abs):
        raise ValueError('Degenerate element: zero or invalid Jacobian.')
    dN_dx = dN_dxi / J
    K = np.zeros((2, 2), dtype=float)
    for w in weights:
        K += E * A * np.outer(dN_dx, dN_dx) * J_abs * w
    return K