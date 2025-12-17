def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 3.2e-06
    Iz = 5.4e-06
    J = 1.1e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    evals = np.linalg.eigvalsh(K)
    max_ev = float(np.max(np.abs(evals)))
    tol = max(1.0, max_ev) * 1e-09
    zero_mask = np.isclose(evals, 0.0, atol=tol)
    assert np.sum(zero_mask) == 6
    positive_evals = evals[~zero_mask]
    assert positive_evals.min() > 0
    EA_L = E * A / L
    G = E / (2 * (1 + nu))
    GJ_L = G * J / L
    K_ax_expected = EA_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_ax = K[np.ix_([0, 6], [0, 6])]
    assert np.allclose(K_ax, K_ax_expected, rtol=1e-12, atol=1e-12)
    K_tor_expected = GJ_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_tor = K[np.ix_([3, 9], [3, 9])]
    assert np.allclose(K_tor, K_tor_expected, rtol=1e-12, atol=1e-12)

    def bending_block(EI):
        return np.array([[12 * EI / L ** 3, 6 * EI / L ** 2, -12 * EI / L ** 3, 6 * EI / L ** 2], [6 * EI / L ** 2, 4 * EI / L, -6 * EI / L ** 2, 2 * EI / L], [-12 * EI / L ** 3, -6 * EI / L ** 2, 12 * EI / L ** 3, -6 * EI / L ** 2], [6 * EI / L ** 2, 2 * EI / L, -6 * EI / L ** 2, 4 * EI / L]])
    idx_w = [2, 4, 8, 10]
    K_bend_y_expected = bending_block(E * Iy)
    K_bend_y = K[np.ix_(idx_w, idx_w)]
    assert np.allclose(K_bend_y, K_bend_y_expected, rtol=1e-12, atol=1e-12)
    idx_v = [1, 5, 7, 11]
    K_bend_z_expected = bending_block(E * Iz)
    K_bend_z = K[np.ix_(idx_v, idx_v)]
    assert np.allclose(K_bend_z, K_bend_z_expected, rtol=1e-12, atol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 200000000000.0
    nu = 0.29
    A = 0.008
    L = 3.0
    Iy = 7.5e-06
    Iz = 5e-06
    J = 1.2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    fixed = [0, 1, 2, 3, 4, 5]
    free = [6, 7, 8, 9, 10, 11]
    Kff = K[np.ix_(free, free)]
    Pz = 1250.0
    F_free = np.zeros(6)
    F_free[2] = Pz
    u_free = np.linalg.solve(Kff, F_free)
    w_tip = u_free[2]
    w_expected = Pz * L ** 3 / (3 * E * Iy)
    assert np.isclose(w_tip, w_expected, rtol=1e-08, atol=1e-12)
    Py = 980.0
    F_free = np.zeros(6)
    F_free[1] = Py
    u_free = np.linalg.solve(Kff, F_free)
    v_tip = u_free[1]
    v_expected = Py * L ** 3 / (3 * E * Iz)
    assert np.isclose(v_tip, v_expected, rtol=1e-08, atol=1e-12)
    Px = 5000.0
    F_free = np.zeros(6)
    F_free[0] = Px
    u_free = np.linalg.solve(Kff, F_free)
    u_tip = u_free[0]
    u_expected = Px * L / (E * A)
    assert np.isclose(u_tip, u_expected, rtol=1e-12, atol=1e-15)