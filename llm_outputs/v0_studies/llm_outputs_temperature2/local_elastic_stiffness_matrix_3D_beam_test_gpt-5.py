def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.02
    L = 3.0
    Iy = 4e-06
    Iz = 5e-06
    J = 6e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    s = max(np.linalg.norm(K, ord=np.inf), 1.0)
    assert np.allclose(K, K.T, atol=1e-12 * s, rtol=1e-12)
    evals = np.linalg.eigvalsh(K)
    tol_zero = 1e-09 * s
    num_near_zero = np.count_nonzero(np.abs(evals) < tol_zero)
    assert num_near_zero >= 6
    k_ax = E * A / L
    axial_idx = [0, 6]
    K_axial = K[np.ix_(axial_idx, axial_idx)]
    K_axial_exp = k_ax * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_axial, K_axial_exp, atol=1e-12 * s, rtol=1e-12)
    G = E / (2.0 * (1.0 + nu))
    k_t = G * J / L
    tors_idx = [3, 9]
    K_tors = K[np.ix_(tors_idx, tors_idx)]
    K_tors_exp = k_t * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_tors, K_tors_exp, atol=1e-12 * s, rtol=1e-12)

    def bending_block(EI):
        L2 = L * L
        L3 = L2 * L
        b = EI / L3
        return b * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    bend_y_idx = [2, 4, 8, 10]
    K_bend_y = K[np.ix_(bend_y_idx, bend_y_idx)]
    K_bend_y_exp = bending_block(E * Iy)
    assert np.allclose(K_bend_y, K_bend_y_exp, atol=1e-12 * s, rtol=1e-12)
    bend_z_idx = [1, 5, 7, 11]
    K_bend_z = K[np.ix_(bend_z_idx, bend_z_idx)]
    K_bend_z_exp = bending_block(E * Iz)
    assert np.allclose(K_bend_z, K_bend_z_exp, atol=1e-12 * s, rtol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.015
    L = 2.0
    Iy = 8e-06
    Iz = 6e-06
    J = 7.5e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_ff = K[6:, 6:]
    assert K_ff.shape == (6, 6)
    Pz = 1000.0
    fz = np.zeros(6)
    fz[2] = Pz
    uz = np.linalg.solve(K_ff, fz)
    w_tip = uz[2]
    w_expected = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w_tip, w_expected, rtol=1e-08, atol=1e-12 * max(abs(w_expected), 1.0))
    Py = 1000.0
    fy = np.zeros(6)
    fy[1] = Py
    uy = np.linalg.solve(K_ff, fy)
    v_tip = uy[1]
    v_expected = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v_tip, v_expected, rtol=1e-08, atol=1e-12 * max(abs(v_expected), 1.0))
    Px = 1000.0
    fx = np.zeros(6)
    fx[0] = Px
    ux = np.linalg.solve(K_ff, fx)
    u_tip = ux[0]
    u_expected = Px * L / (E * A)
    assert np.isclose(u_tip, u_expected, rtol=1e-10, atol=1e-12 * max(abs(u_expected), 1.0))