def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    (E, nu, A, L, Iy, Iz, J) = (200000000000.0, 0.3, 0.1, 2.0, 0.0001, 0.0001, 1e-05)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix must be 12x12'
    assert np.allclose(K, K.T), 'Stiffness matrix must be symmetric'
    eigenvalues = np.linalg.eigvals(K)
    zero_eigenvalues = np.sum(np.isclose(eigenvalues, 0, atol=1e-10))
    assert zero_eigenvalues == 6, 'Stiffness matrix must have 6 zero eigenvalues for rigid body modes'
    k_axial = E * A / L
    assert np.isclose(K[0, 0], k_axial), 'Axial stiffness term (u1) incorrect'
    assert np.isclose(K[6, 6], k_axial), 'Axial stiffness term (u2) incorrect'
    assert np.isclose(K[0, 6], -k_axial), 'Axial stiffness coupling term incorrect'
    G = E / (2 * (1 + nu))
    k_torsion = G * J / L
    assert np.isclose(K[3, 3], k_torsion), 'Torsional stiffness term (θx1) incorrect'
    assert np.isclose(K[9, 9], k_torsion), 'Torsional stiffness term (θx2) incorrect'
    assert np.isclose(K[3, 9], -k_torsion), 'Torsional stiffness coupling term incorrect'
    k_bend_z_11 = 12 * E * Iz / L ** 3
    k_bend_z_12 = 6 * E * Iz / L ** 2
    k_bend_z_22 = 4 * E * Iz / L
    assert np.isclose(K[1, 1], k_bend_z_11), 'Bending stiffness (z-dir) term (v1) incorrect'
    assert np.isclose(K[1, 5], k_bend_z_12), 'Bending stiffness (z-dir) coupling term incorrect'
    assert np.isclose(K[5, 5], k_bend_z_22), 'Bending stiffness (z-dir) term (θz1) incorrect'
    k_bend_y_11 = 12 * E * Iy / L ** 3
    k_bend_y_12 = -6 * E * Iy / L ** 2
    k_bend_y_22 = 4 * E * Iy / L
    assert np.isclose(K[2, 2], k_bend_y_11), 'Bending stiffness (y-dir) term (w1) incorrect'
    assert np.isclose(K[2, 4], k_bend_y_12), 'Bending stiffness (y-dir) coupling term incorrect'
    assert np.isclose(K[4, 4], k_bend_y_22), 'Bending stiffness (y-dir) term (θy1) incorrect'

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    (E, nu, A, L, Iy, Iz, J) = (200000000000.0, 0.3, 0.01, 1.0, 1e-05, 1e-05, 1e-06)
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_reduced = K[6:, 6:]
    Fz = 1000.0
    F = np.array([0, 0, Fz, 0, 0, 0])
    u = np.linalg.solve(K_reduced, F)
    w_computed = u[2]
    w_analytical = Fz * L ** 3 / (3 * E * Iy)
    assert np.isclose(w_computed, w_analytical), 'Z-direction deflection does not match Euler-Bernoulli theory'
    Fy = 1000.0
    F = np.array([0, Fy, 0, 0, 0, 0])
    u = np.linalg.solve(K_reduced, F)
    v_computed = u[1]
    v_analytical = Fy * L ** 3 / (3 * E * Iz)
    assert np.isclose(v_computed, v_analytical), 'Y-direction deflection does not match Euler-Bernoulli theory'
    Fx = 1000.0
    F = np.array([Fx, 0, 0, 0, 0, 0])
    u = np.linalg.solve(K_reduced, F)
    u_computed = u[0]
    u_analytical = Fx * L / (E * A)
    assert np.isclose(u_computed, u_analytical), 'X-direction deflection does not match Euler-Bernoulli theory'