def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.0032
    L = 2.5
    Iy = 1.1e-06
    Iz = 2.3e-06
    J = 9e-07
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-12, rtol=0)
    evals = np.linalg.eigvalsh(0.5 * (K + K.T))
    max_eval = np.max(np.abs(evals))
    tol_eval = max(1e-12, max_eval * 1e-09)
    zeros = np.sum(evals <= tol_eval)
    assert zeros == 6
    G = E / (2.0 * (1.0 + nu))
    idx_u = [0, 6]
    K_uu = K[np.ix_(idx_u, idx_u)]
    K_uu_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_uu, K_uu_expected, rtol=1e-12, atol=1e-10)
    idx_tx = [3, 9]
    K_tx = K[np.ix_(idx_tx, idx_tx)]
    K_tx_expected = G * J / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_tx, K_tx_expected, rtol=1e-12, atol=1e-10)
    idx_w_thetay = [2, 4, 8, 10]
    K_wy = K[np.ix_(idx_w_thetay, idx_w_thetay)]
    L2 = L * L
    by_mat = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    K_wy_expected = E * Iy / L ** 3 * by_mat
    assert np.allclose(K_wy, K_wy_expected, rtol=1e-12, atol=1e-08)
    idx_v_thetaz = [1, 5, 7, 11]
    K_vz = K[np.ix_(idx_v_thetaz, idx_v_thetaz)]
    bz_mat = by_mat
    K_vz_expected = E * Iz / L ** 3 * bz_mat
    assert np.allclose(K_vz, K_vz_expected, rtol=1e-12, atol=1e-08)
    other_idx = [i for i in range(12) if i not in idx_u]
    assert np.allclose(K[np.ix_(idx_u, other_idx)], 0.0, atol=1e-10)
    assert np.allclose(K[np.ix_(other_idx, idx_u)], 0.0, atol=1e-10)
    idx_w_plane = idx_w_thetay
    idx_v_plane = idx_v_thetaz
    assert np.allclose(K[np.ix_(idx_w_plane, idx_v_plane)], 0.0, atol=1e-10)
    assert np.allclose(K[np.ix_(idx_v_plane, idx_w_plane)], 0.0, atol=1e-10)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory."""
    E = 70000000000.0
    nu = 0.33
    A = 0.0025
    L = 1.75
    Iy = 8e-07
    Iz = 1.6e-06
    J = 5e-07
    K = fcn(E, nu, A, L, Iy, Iz, J)
    fixed = [0, 1, 2, 3, 4, 5]
    free = [6, 7, 8, 9, 10, 11]
    K_ff = K[np.ix_(free, free)]
    Pz = 1234.5
    F = np.zeros(12)
    F[8] = Pz
    Ff = F[free]
    d = np.linalg.solve(K_ff, Ff)
    w_tip = d[2]
    w_analytical = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w_tip, w_analytical, rtol=1e-10, atol=1e-14)
    Py = 2345.6
    F[:] = 0.0
    F[7] = Py
    Ff = F[free]
    d = np.linalg.solve(K_ff, Ff)
    v_tip = d[1]
    v_analytical = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v_tip, v_analytical, rtol=1e-10, atol=1e-14)
    Px = 3456.7
    F[:] = 0.0
    F[6] = Px
    Ff = F[free]
    d = np.linalg.solve(K_ff, Ff)
    u_tip = d[0]
    u_analytical = Px * L / (E * A)
    assert np.isclose(u_tip, u_analytical, rtol=1e-12, atol=1e-14)