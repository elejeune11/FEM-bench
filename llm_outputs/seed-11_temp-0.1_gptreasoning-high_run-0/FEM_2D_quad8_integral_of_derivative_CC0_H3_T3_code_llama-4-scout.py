def FEM_2D_quad8_integral_of_derivative_CC0_H3_T3(node_coords: np.ndarray, node_values: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    """
    Compute ∫_Ω (∇u) dΩ for a scalar field u defined over a quadratic
    8-node quadrilateral (Q8) finite element.
    The computation uses isoparametric mapping and Gauss–Legendre quadrature
    on the reference domain Q = [-1, 1] × [-1, 1].
    Parameters
    ----------
    node_coords : np.ndarray
        Physical coordinates of the Q8 element nodes.
        Shape: (8, 2). Each row is [x, y].
        Node ordering (must match both geometry and values):
            1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
            5: ( 0, -1),  6: ( 1,  0),  7: ( 0, 1),  8: (-1,  0)
    node_values : np.ndarray
        Scalar nodal values of u. Shape: (8,) or (8, 1).
    num_gauss_pts : int
        Number of quadrature points to use: one of {1, 4, 9}.
    Returns
    -------
    integral : np.ndarray
        The vector [∫_Ω ∂u/∂x dΩ, ∫_Ω ∂u/∂y dΩ].
        Shape: (2,).
    Notes
    -----
    Shape functions:
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    """

    def N(ξ, η, i):
        if i == 0:
            return -1 / 4 * (1 - ξ) * (1 - η) * (1 + ξ + η)
        elif i == 1:
            return 1 / 4 * (1 + ξ) * (1 - η) * (ξ - η - 1)
        elif i == 2:
            return 1 / 4 * (1 + ξ) * (1 + η) * (ξ + η - 1)
        elif i == 3:
            return 1 / 4 * (1 - ξ) * (1 + η) * (η - ξ - 1)
        elif i == 4:
            return 1 / 2 * (1 - ξ ** 2) * (1 - η)
        elif i == 5:
            return 1 / 2 * (1 + ξ) * (1 - η ** 2)
        elif i == 6:
            return 1 / 2 * (1 - ξ ** 2) * (1 + η)
        elif i == 7:
            return 1 / 2 * (1 - ξ) * (1 - η ** 2)

    def dN_dξ(ξ, η, i):
        if i == 0:
            return -1 / 4 * ((1 - η) * (1 + ξ + η) + (1 - ξ) * (1 - η))
        elif i == 1:
            return 1 / 4 * ((1 - η) * (ξ - η - 1) + (1 + ξ) * (1 - η))
        elif i == 2:
            return 1 / 4 * ((1 + η) * (ξ + η - 1) + (1 + ξ) * (1 + η))
        elif i == 3:
            return 1 / 4 * (-(1 + η) * (η - ξ - 1) - (1 - ξ) * (1 + η))
        elif i == 4:
            return -ξ * (1 - η)
        elif i == 5:
            return 1 / 2 * (1 - η ** 2)
        elif i == 6:
            return -ξ * (1 + η)
        elif i == 7:
            return -1 / 2 * (1 - η ** 2)

    def dN_dη(ξ, η, i):
        if i == 0:
            return -1 / 4 * ((1 - ξ) * (1 + ξ + η) + (1 - ξ) * (1 - η))
        elif i == 1:
            return -1 / 4 * (1 + ξ) * (ξ - η - 1) + 1 / 4 * (1 + ξ) * (1 - η)
        elif i == 2:
            return 1 / 4 * ((1 + ξ) * (ξ + η - 1) + (1 + ξ) * (1 + η))
        elif i == 3:
            return 1 / 4 * ((1 - ξ) * (η - ξ - 1) + (1 - ξ) * (1 + η))
        elif i == 4:
            return -1 / 2 * (1 - ξ ** 2)
        elif i == 5:
            return -η * (1 + ξ)
        elif i == 6:
            return 1 / 2 * (1 - ξ ** 2)
        elif i == 7:
            return η * (1 - ξ)
    if num_gauss_pts == 1:
        gauss_pts = [(0, 0)]
        gauss_weights = [4]
    elif num_gauss_pts == 4:
        gauss_pts = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]
        gauss_weights = [1, 1, 1, 1]
    elif num_gauss_pts == 9:
        gauss_pts = [(-np.sqrt(3 / 5), -np.sqrt(3 / 5)), (np.sqrt(3 / 5), -np.sqrt(3 / 5)), (0, -np.sqrt(3 / 5)), (-np.sqrt(3 / 5), np.sqrt(3 / 5)), (np.sqrt(3 / 5), np.sqrt(3 / 5)), (0, np.sqrt(3 / 5)), (-np.sqrt(3 / 5), 0), (np.sqrt(3 / 5), 0), (0, 0)]
        gauss_weights = [25 / 81, 25 / 81, 40 / 81, 25 / 81, 25 / 81, 40 / 81, 40 / 81, 40 / 81, 64 / 81]

    def jacobian(ξ, η):
        dx_dξ = 0
        dx_dη = 0
        dy_dξ = 0
        dy_dη = 0
        for i in range(8):
            dx_dξ += node_coords[i, 0] * dN_dξ(ξ, η, i)
            dx_dη += node_coords[i, 0] * dN_dη(ξ, η, i)
            dy_dξ += node_coords[i, 1] * dN_dξ(ξ, η, i)
            dy_dη += node_coords[i, 1] * dN_dη(ξ, η, i)
        det_J = dx_dξ * dy_dη - dx_dη * dy_dξ
        return np.array([[dy_dη, -dx_dη], [-dy_dξ, dx_dξ]]) / det_J
    integral = np.zeros(2)
    for (gauss_pt, weight) in zip(gauss_pts, gauss_weights):
        (ξ, η) = gauss_pt
        J_inv = jacobian(ξ, η)
        dN_dx = np.zeros(8)
        dN_dy = np.zeros(8)
        for i in range(8):
            dN_dξ_i = dN_dξ(ξ, η, i)
            dN_dη_i = dN_dη(ξ, η, i)
            dN_dx[i] = J_inv[0, 0] * dN_dξ_i + J_inv[0, 1] * dN_dη_i
            dN_dy[i] = J_inv[1, 0] * dN_dξ_i + J_inv[1, 1] * dN_dη_i
        u = np.sum(node_values * np.array([N(ξ, η, i) for i in range(8)]))
        integral[0] += weight * np.sum(node_values * dN_dx) * det_jacobian(ξ, η)
        integral[1] += weight * np.sum(node_values * dN_dy) * det_jacobian(ξ, η)
    return integral