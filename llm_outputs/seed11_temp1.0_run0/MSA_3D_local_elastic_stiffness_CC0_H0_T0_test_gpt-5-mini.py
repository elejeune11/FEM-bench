def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1.5e-06
    Iz = 2.5e-06
    J = 8e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-12, rtol=0)
    eigs = np.linalg.eigvalsh(K)
    max_abs = np.max(np.abs(eigs))
    tol = max_abs * 1e-08 if max_abs > 0 else 1e-12
    zero_count = int(np.sum(np.abs(eigs) <= tol))
    assert zero_count >= 6
    k_axial = E * A / L
    assert np.isclose(K[0, 0], k_axial, rtol=1e-08, atol=0)
    assert np.isclose(K[6, 6], k_axial, rtol=1e-08, atol=0)
    assert np.isclose(K[0, 6], -k_axial, rtol=1e-08, atol=0)
    assert np.isclose(K[6, 0], -k_axial, rtol=1e-08, atol=0)
    G = E / (2.0 * (1.0 + nu))
    k_torsion = G * J / L
    assert np.isclose(K[3, 3], k_torsion, rtol=1e-08, atol=0)
    assert np.isclose(K[9, 9], k_torsion, rtol=1e-08, atol=0)
    assert np.isclose(K[3, 9], -k_torsion, rtol=1e-08, atol=0)
    assert np.isclose(K[9, 3], -k_torsion, rtol=1e-08, atol=0)
    idx_w = [2, 4, 8, 10]
    B_coeff = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L]], dtype=float)
    K_bend_y_expected = E * Iy / L ** 3 * B_coeff
    K_bend_y_actual = K[np.ix_(idx_w, idx_w)]
    assert np.allclose(K_bend_y_actual, K_bend_y_expected, rtol=1e-08, atol=0)
    idx_v = [1, 5, 7, 11]
    K_bend_z_expected = E * Iz / L ** 3 * B_coeff
    K_bend_z_actual = K[np.ix_(idx_v, idx_v)]
    assert np.allclose(K_bend_z_actual, K_bend_z_expected, rtol=1e-08, atol=0)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply point loads at the tip of a cantilever beam and verify displacements:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.005
    L = 1.5
    Iy = 2e-06
    Iz = 3e-06
    J = 1e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    free_idx = list(range(6, 12))
    Kf = K[np.ix_(free_idx, free_idx)]
    Pz = 1000.0
    f_reduced = np.zeros(6)
    f_reduced[2] = Pz
    u = np.linalg.solve(Kf, f_reduced)
    uz_tip = u[2]
    uz_expected = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(uz_tip, uz_expected, rtol=1e-06, atol=1e-12)
    Py = 800.0
    f_reduced = np.zeros(6)
    f_reduced[1] = Py
    u = np.linalg.solve(Kf, f_reduced)
    uy_tip = u[1]
    uy_expected = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(uy_tip, uy_expected, rtol=1e-06, atol=1e-12)
    Px = 2000.0
    f_reduced = np.zeros(6)
    f_reduced[0] = Px
    u = np.linalg.solve(Kf, f_reduced)
    ux_tip = u[0]
    ux_expected = Px * L / (E * A)
    assert np.isclose(ux_tip, ux_expected, rtol=1e-06, atol=1e-12)