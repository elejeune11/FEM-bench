def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.333e-06
    Iz = 1.25e-05
    J = 2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    ev = np.linalg.eigvalsh((K + K.T) / 2.0)
    max_ev = np.max(np.abs(ev))
    tol_ev = max(1e-12, 1e-09 * max_ev)
    assert np.all(ev >= -tol_ev)
    num_zero = np.sum(np.abs(ev) <= tol_ev)
    assert num_zero == 6
    EA_L = E * A / L
    axial_idx = [0, 6]
    K_axial = K[np.ix_(axial_idx, axial_idx)]
    K_axial_exp = EA_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_axial, K_axial_exp, rtol=1e-12, atol=1e-12)
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    tors_idx = [3, 9]
    K_tors = K[np.ix_(tors_idx, tors_idx)]
    K_tors_exp = GJ_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_tors, K_tors_exp, rtol=1e-12, atol=1e-12)

    def bending_block(I, dof_idx):
        L2 = L * L
        c1 = 12.0 * E * I / L ** 3
        c2 = 6.0 * E * I / L ** 2
        c3 = 4.0 * E * I / L
        c4 = 2.0 * E * I / L
        Kb = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]])
        return (K[np.ix_(dof_idx, dof_idx)], Kb)
    (K_bend_y, K_bend_y_exp) = bending_block(Iy, [2, 4, 8, 10])
    assert np.allclose(K_bend_y, K_bend_y_exp, rtol=1e-12, atol=1e-12)
    (K_bend_z, K_bend_z_exp) = bending_block(Iz, [1, 5, 7, 11])
    assert np.allclose(K_bend_z, K_bend_z_exp, rtol=1e-12, atol=1e-12)
    other_idx = list(set(range(12)) - set(axial_idx))
    assert np.allclose(K[np.ix_(axial_idx, other_idx)], 0.0, atol=1e-10)
    other_idx = list(set(range(12)) - set(tors_idx))
    assert np.allclose(K[np.ix_(tors_idx, other_idx)], 0.0, atol=1e-10)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.333e-06
    Iz = 1.25e-05
    J = 2e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    fixed = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    free = np.array([6, 7, 8, 9, 10, 11], dtype=int)
    Kff = K[np.ix_(free, free)]
    inv_tol = 1e-12
    assert np.linalg.cond(Kff) < 1.0 / inv_tol
    free_index_map = {d: i for (i, d) in enumerate(free)}
    Pz = 1000.0
    F = np.zeros(12)
    F[8] = Pz
    Uf = np.linalg.solve(Kff, F[free])
    w2 = Uf[free_index_map[8]]
    w2_expected = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w2, w2_expected, rtol=1e-09, atol=1e-12)
    Py = 1000.0
    F = np.zeros(12)
    F[7] = Py
    Uf = np.linalg.solve(Kff, F[free])
    v2 = Uf[free_index_map[7]]
    v2_expected = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v2, v2_expected, rtol=1e-09, atol=1e-12)
    Px = 1000.0
    F = np.zeros(12)
    F[6] = Px
    Uf = np.linalg.solve(Kff, F[free])
    u2 = Uf[free_index_map[6]]
    u2_expected = Px * L / (E * A)
    assert np.isclose(u2, u2_expected, rtol=1e-12, atol=1e-12)