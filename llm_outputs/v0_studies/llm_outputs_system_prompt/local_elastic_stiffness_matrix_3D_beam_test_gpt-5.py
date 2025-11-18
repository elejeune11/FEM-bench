def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    np = fcn.__globals__['np']
    E = 210000000000.0
    nu = 0.28
    A = 0.008
    L = 3.2
    Iy = 7e-06
    Iz = 5.5e-06
    J = 1.3e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    scale = max(1.0, float(np.max(np.abs(K))))
    assert np.allclose(K, K.T, atol=1e-12 * scale, rtol=0.0)
    axial_idx = [0, 6]
    K_ax = K[np.ix_(axial_idx, axial_idx)]
    K_ax_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    ax_scale = max(1.0, float(np.max(np.abs(K_ax_expected))))
    assert np.allclose(K_ax, K_ax_expected, atol=1e-12 * ax_scale, rtol=0.0)
    tors_idx = [3, 9]
    K_tors = K[np.ix_(tors_idx, tors_idx)]
    G = E / (2.0 * (1.0 + nu))
    K_tors_expected = G * J / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    tors_scale = max(1.0, float(np.max(np.abs(K_tors_expected))))
    assert np.allclose(K_tors, K_tors_expected, atol=1e-12 * tors_scale, rtol=0.0)
    bend_y_idx = [2, 4, 8, 10]
    K_by = K[np.ix_(bend_y_idx, bend_y_idx)]
    L2 = L * L
    L3 = L2 * L
    Mb = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
    K_by_expected = E * Iy / L3 * Mb
    by_scale = max(1.0, float(np.max(np.abs(K_by_expected))))
    assert np.allclose(K_by, K_by_expected, atol=1e-11 * by_scale, rtol=0.0)
    bend_z_idx = [1, 5, 7, 11]
    K_bz = K[np.ix_(bend_z_idx, bend_z_idx)]
    K_bz_expected = E * Iz / L3 * Mb
    bz_scale = max(1.0, float(np.max(np.abs(K_bz_expected))))
    assert np.allclose(K_bz, K_bz_expected, atol=1e-11 * bz_scale, rtol=0.0)
    other_ax = [i for i in range(12) if i not in axial_idx]
    assert np.allclose(K[np.ix_(axial_idx, other_ax)], 0.0, atol=1e-11 * scale, rtol=0.0)
    assert np.allclose(K[np.ix_(other_ax, axial_idx)], 0.0, atol=1e-11 * scale, rtol=0.0)
    other_tors = [i for i in range(12) if i not in tors_idx]
    assert np.allclose(K[np.ix_(tors_idx, other_tors)], 0.0, atol=1e-11 * scale, rtol=0.0)
    assert np.allclose(K[np.ix_(other_tors, tors_idx)], 0.0, atol=1e-11 * scale, rtol=0.0)
    assert np.allclose(K[np.ix_(bend_y_idx, bend_z_idx)], 0.0, atol=1e-11 * scale, rtol=0.0)
    assert np.allclose(K[np.ix_(bend_z_idx, bend_y_idx)], 0.0, atol=1e-11 * scale, rtol=0.0)

    def check_null(vec):
        res = K @ vec
        tol = 1e-08 * scale * max(1.0, float(np.linalg.norm(vec)))
        assert float(np.linalg.norm(res)) <= tol
    v_rbm = np.zeros(12, dtype=float)
    v_rbm[0] = 1.0
    v_rbm[6] = 1.0
    check_null(v_rbm)
    v_rbm = np.zeros(12, dtype=float)
    v_rbm[1] = 1.0
    v_rbm[7] = 1.0
    check_null(v_rbm)
    v_rbm = np.zeros(12, dtype=float)
    v_rbm[2] = 1.0
    v_rbm[8] = 1.0
    check_null(v_rbm)
    v_rbm = np.zeros(12, dtype=float)
    v_rbm[3] = 1.0
    v_rbm[9] = 1.0
    check_null(v_rbm)
    v_rbm = np.zeros(12, dtype=float)
    v_rbm[4] = 1.0
    v_rbm[8] = L
    v_rbm[10] = 1.0
    check_null(v_rbm)
    v_rbm = np.zeros(12, dtype=float)
    v_rbm[5] = 1.0
    v_rbm[7] = L
    v_rbm[11] = 1.0
    check_null(v_rbm)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply point loads at the tip of a cantilever beam and verify that tip displacements match Euler-Bernoulli beam theory:
    """
    np = fcn.__globals__['np']
    E = 210000000000.0
    nu = 0.3
    A = 0.003
    L = 2.0
    Iy = 4e-06
    Iz = 6e-06
    J = 1.5e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    free = [6, 7, 8, 9, 10, 11]
    K_ff = K[np.ix_(free, free)]
    Pz = 1234.5
    fz = np.zeros(6, dtype=float)
    fz[2] = Pz
    uz = np.linalg.solve(K_ff, fz)
    w_tip_expected = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(uz[2], w_tip_expected, rtol=1e-09, atol=1e-14)
    Py = 789.0
    fy = np.zeros(6, dtype=float)
    fy[1] = Py
    uy = np.linalg.solve(K_ff, fy)
    v_tip_expected = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(uy[1], v_tip_expected, rtol=1e-09, atol=1e-14)
    Px = 456.0
    fx = np.zeros(6, dtype=float)
    fx[0] = Px
    ux = np.linalg.solve(K_ff, fx)
    u_tip_expected = Px * L / (E * A)
    assert np.isclose(ux[0], u_tip_expected, rtol=1e-12, atol=1e-18)