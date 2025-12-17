def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    L = 2.5
    Iy = 8e-06
    Iz = 5e-06
    J = 1.2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    kmax = np.max(np.abs(K))
    eigvals = np.linalg.eigvalsh((K + K.T) / 2.0)
    zero_tol = 1e-08 * max(kmax, 1.0)
    num_zero = int(np.sum(np.abs(eigvals) < zero_tol))
    assert num_zero == 6
    idx_u = [0, 6]
    Kuu = K[np.ix_(idx_u, idx_u)]
    Kuu_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(Kuu, Kuu_expected, rtol=1e-12, atol=1e-12)
    other_idx = [i for i in range(12) if i not in idx_u]
    assert np.allclose(K[np.ix_(idx_u, other_idx)], 0.0, rtol=0, atol=1e-12)
    idx_tx = [3, 9]
    Ktt = K[np.ix_(idx_tx, idx_tx)]
    G = E / (2.0 * (1.0 + nu))
    Ktt_expected = G * J / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(Ktt, Ktt_expected, rtol=1e-12, atol=1e-12)
    other_idx = [i for i in range(12) if i not in idx_tx]
    assert np.allclose(K[np.ix_(idx_tx, other_idx)], 0.0, rtol=0, atol=1e-12)
    idx_w_thy = [2, 4, 8, 10]
    K_wthy = K[np.ix_(idx_w_thy, idx_w_thy)]
    ky = E * Iy / L ** 3
    K_wthy_expected = np.array([[12 * ky, 6 * L * ky, -12 * ky, 6 * L * ky], [6 * L * ky, 4 * L * L * ky, -6 * L * ky, 2 * L * L * ky], [-12 * ky, -6 * L * ky, 12 * ky, -6 * L * ky], [6 * L * ky, 2 * L * L * ky, -6 * L * ky, 4 * L * L * ky]])
    assert np.allclose(K_wthy, K_wthy_expected, rtol=1e-12, atol=1e-09)
    idx_v_thz = [1, 5, 7, 11]
    K_vthz = K[np.ix_(idx_v_thz, idx_v_thz)]
    kz = E * Iz / L ** 3
    K_vthz_expected = np.array([[12 * kz, 6 * L * kz, -12 * kz, 6 * L * kz], [6 * L * kz, 4 * L * L * kz, -6 * L * kz, 2 * L * L * kz], [-12 * kz, -6 * L * kz, 12 * kz, -6 * L * kz], [6 * L * kz, 2 * L * L * kz, -6 * L * kz, 4 * L * L * kz]])
    assert np.allclose(K_vthz, K_vthz_expected, rtol=1e-12, atol=1e-09)
    assert np.allclose(K[np.ix_(idx_w_thy, idx_v_thz)], 0.0, rtol=0, atol=1e-12)
    assert np.allclose(K[np.ix_(idx_v_thz, idx_w_thy)], 0.0, rtol=0, atol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply tip loads to a cantilever beam (clamped at node 1, free at node 2) and verify tip displacements:
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.003
    L = 2.0
    Iy = 4e-06
    Iz = 7e-06
    J = 5e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    free = np.array([6, 7, 8, 9, 10, 11], dtype=int)
    Kff = K[np.ix_(free, free)]
    Pz = 1000.0
    Ff_z = np.zeros(6)
    Ff_z[2] = Pz
    d_free_z = np.linalg.solve(Kff, Ff_z)
    w2 = d_free_z[2]
    w2_expected = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w2, w2_expected, rtol=1e-09, atol=0.0)
    Py = 1000.0
    Ff_y = np.zeros(6)
    Ff_y[1] = Py
    d_free_y = np.linalg.solve(Kff, Ff_y)
    v2 = d_free_y[1]
    v2_expected = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v2, v2_expected, rtol=1e-09, atol=0.0)
    Px = 100000.0
    Ff_x = np.zeros(6)
    Ff_x[0] = Px
    d_free_x = np.linalg.solve(Kff, Ff_x)
    u2 = d_free_x[0]
    u2_expected = Px * L / (E * A)
    assert np.isclose(u2, u2_expected, rtol=1e-12, atol=0.0)