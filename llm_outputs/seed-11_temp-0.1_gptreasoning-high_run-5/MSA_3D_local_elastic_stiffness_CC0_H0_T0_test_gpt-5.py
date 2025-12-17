def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.012
    L = 2.4
    Iy = 8e-06
    Iz = 5e-06
    J = 2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    scale = float(np.max(np.abs(K))) if K.size else 1.0
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12 * max(1.0, scale))
    s = np.linalg.svd(K, compute_uv=False)
    smax = s.max() if s.size else 1.0
    num_small = int(np.sum(s / smax < 1e-11))
    assert num_small == 6
    k_ax = E * A / L
    axial_idx = [0, 6]
    K_ax = K[np.ix_(axial_idx, axial_idx)]
    exp_ax = k_ax * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_ax, exp_ax, rtol=1e-12, atol=1e-12 * max(1.0, k_ax))
    off_ax = K[np.ix_(axial_idx, [i for i in range(12) if i not in axial_idx])]
    assert np.allclose(off_ax, 0.0, atol=1e-10 * scale)
    G = E / (2.0 * (1.0 + nu))
    k_tor = G * J / L
    torsion_idx = [3, 9]
    K_tor = K[np.ix_(torsion_idx, torsion_idx)]
    exp_tor = k_tor * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_tor, exp_tor, rtol=1e-12, atol=1e-12 * max(1.0, k_tor))
    off_tor = K[np.ix_(torsion_idx, [i for i in range(12) if i not in torsion_idx])]
    assert np.allclose(off_tor, 0.0, atol=1e-10 * scale)
    bz_idx = [1, 5, 7, 11]
    L2 = L ** 2
    fac_z = E * Iz / L ** 3
    exp_bz = fac_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    K_bz = K[np.ix_(bz_idx, bz_idx)]
    assert np.allclose(K_bz, exp_bz, rtol=1e-12, atol=1e-10 * max(1.0, np.max(np.abs(exp_bz))))
    by_idx = [2, 4, 8, 10]
    fac_y = E * Iy / L ** 3
    exp_by = fac_y * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    K_by = K[np.ix_(by_idx, by_idx)]
    assert np.allclose(K_by, exp_by, rtol=1e-12, atol=1e-10 * max(1.0, np.max(np.abs(exp_by))))
    cross_bending = K[np.ix_(bz_idx, by_idx)]
    assert np.allclose(cross_bending, 0.0, atol=1e-10 * scale)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 70000000000.0
    nu = 0.31
    A = 0.005
    L = 2.5
    Iy = 9e-06
    Iz = 6e-06
    J = 0.00012
    K = fcn(E, nu, A, L, Iy, Iz, J)
    Kff = K[6:12, 6:12]
    Pz = 1234.5
    Fz = np.zeros(6)
    Fz[2] = Pz
    d_free_z = np.linalg.solve(Kff, Fz)
    w_tip = d_free_z[2]
    w_expected = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w_tip, w_expected, rtol=1e-09, atol=1e-12 + 1e-12 * abs(w_expected))
    Py = 987.6
    Fy = np.zeros(6)
    Fy[1] = Py
    d_free_y = np.linalg.solve(Kff, Fy)
    v_tip = d_free_y[1]
    v_expected = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v_tip, v_expected, rtol=1e-09, atol=1e-12 + 1e-12 * abs(v_expected))
    Px = 4321.0
    Fx = np.zeros(6)
    Fx[0] = Px
    d_free_x = np.linalg.solve(Kff, Fx)
    u_tip = d_free_x[0]
    u_expected = Px * L / (E * A)
    assert np.isclose(u_tip, u_expected, rtol=1e-09, atol=1e-15 + 1e-12 * abs(u_expected))