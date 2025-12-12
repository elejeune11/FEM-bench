def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8e-06
    Iz = 5e-06
    J = 1e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    ev = np.linalg.eigvalsh(K)
    tol_zero = max(ev[-1], 1.0) * 1e-10
    n_zero = int(np.sum(np.abs(ev) < tol_zero))
    assert n_zero == 6
    assert np.all(ev[ev > tol_zero] > 0.0)
    k_ax = E * A / L
    K_ax_expected = k_ax * np.array([[1.0, -1.0], [-1.0, 1.0]])
    ax_idx = [0, 6]
    assert np.allclose(K[np.ix_(ax_idx, ax_idx)], K_ax_expected, rtol=1e-10, atol=1e-09)
    G = E / (2.0 * (1.0 + nu))
    k_tor = G * J / L
    K_tor_expected = k_tor * np.array([[1.0, -1.0], [-1.0, 1.0]])
    tor_idx = [3, 9]
    assert np.allclose(K[np.ix_(tor_idx, tor_idx)], K_tor_expected, rtol=1e-10, atol=1e-09)
    k_bz = E * Iz / L ** 3
    L2 = L * L
    K_bz_expected = k_bz * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    bz_idx = [1, 5, 7, 11]
    assert np.allclose(K[np.ix_(bz_idx, bz_idx)], K_bz_expected, rtol=1e-10, atol=1e-08)
    k_by = E * Iy / L ** 3
    K_by_expected = k_by * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    by_idx = [2, 4, 8, 10]
    assert np.allclose(K[np.ix_(by_idx, by_idx)], K_by_expected, rtol=1e-10, atol=1e-08)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 3.0
    Iy = 8e-06
    Iz = 5e-06
    J = 1e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    fixed = list(range(6))
    free = list(range(6, 12))
    K_ff = K[np.ix_(free, free)]
    F_mag = 1000.0
    f = np.zeros(12)
    f[8] = F_mag
    u_free = np.linalg.solve(K_ff, f[free])
    u = np.zeros(12)
    u[free] = u_free
    w_tip = u[8]
    w_expected = F_mag * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w_tip, w_expected, rtol=1e-08, atol=1e-12)
    f = np.zeros(12)
    f[7] = F_mag
    u_free = np.linalg.solve(K_ff, f[free])
    u = np.zeros(12)
    u[free] = u_free
    v_tip = u[7]
    v_expected = F_mag * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v_tip, v_expected, rtol=1e-08, atol=1e-12)
    f = np.zeros(12)
    f[6] = F_mag
    u_free = np.linalg.solve(K_ff, f[free])
    u = np.zeros(12)
    u[free] = u_free
    u_tip = u[6]
    u_expected = F_mag * L / (E * A)
    assert np.isclose(u_tip, u_expected, rtol=1e-12, atol=1e-18)