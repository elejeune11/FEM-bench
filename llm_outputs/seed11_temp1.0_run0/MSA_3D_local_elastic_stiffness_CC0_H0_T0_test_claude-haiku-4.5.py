def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 8.333e-05
    Iz = 8.333e-05
    J = 0.0001667
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), f'Expected shape (12, 12), got {K.shape}'
    assert np.allclose(K, K.T, atol=1e-10), 'Stiffness matrix must be symmetric'
    eigenvalues = np.linalg.eigvals(K)
    zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-06)
    assert zero_eigenvalues == 6, f'Expected 6 zero eigenvalues (rigid body modes), got {zero_eigenvalues}'
    EA_L = E * A / L
    assert np.isclose(K[0, 0], EA_L, rtol=1e-06), 'Axial stiffness K[0,0] incorrect'
    assert np.isclose(K[6, 6], EA_L, rtol=1e-06), 'Axial stiffness K[6,6] incorrect'
    assert np.isclose(K[0, 6], -EA_L, rtol=1e-06), 'Axial coupling K[0,6] incorrect'
    GJ_L = E / (2 * (1 + nu)) * J / L
    assert np.isclose(K[3, 3], GJ_L, rtol=1e-06), 'Torsional stiffness K[3,3] incorrect'
    assert np.isclose(K[9, 9], GJ_L, rtol=1e-06), 'Torsional stiffness K[9,9] incorrect'
    assert np.isclose(K[3, 9], -GJ_L, rtol=1e-06), 'Torsional coupling K[3,9] incorrect'
    EI_L3 = E * Iy / L ** 3
    assert np.isclose(K[1, 1], 12 * EI_L3, rtol=1e-06), 'Bending y stiffness K[1,1] incorrect'
    assert np.isclose(K[2, 2], 12 * E * Iz / L ** 3, rtol=1e-06), 'Bending z stiffness K[2,2] incorrect'

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply perpendicular and parallel point loads to the tip of a cantilever beam
    and verify that the computed displacement matches the analytical solution
    from Euler-Bernoulli beam theory.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 8.333e-05
    Iz = 8.333e-05
    J = 0.0001667
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_condensed = K[6:12, 6:12]
    F_z = 1000.0
    deflection_z_analytical = F_z * L ** 3 / (3 * E * Iz)
    F_z_vector = np.zeros(6)
    F_z_vector[2] = F_z
    u_z = np.linalg.solve(K_condensed, F_z_vector)
    deflection_z_computed = u_z[2]
    assert np.isclose(deflection_z_computed, deflection_z_analytical, rtol=1e-06), f'Z-direction deflection mismatch: computed={deflection_z_computed}, analytical={deflection_z_analytical}'
    F_y = 1000.0
    deflection_y_analytical = F_y * L ** 3 / (3 * E * Iy)
    F_y_vector = np.zeros(6)
    F_y_vector[1] = F_y
    u_y = np.linalg.solve(K_condensed, F_y_vector)
    deflection_y_computed = u_y[1]
    assert np.isclose(deflection_y_computed, deflection_y_analytical, rtol=1e-06), f'Y-direction deflection mismatch: computed={deflection_y_computed}, analytical={deflection_y_analytical}'
    F_x = 10000.0
    deflection_x_analytical = F_x * L / (E * A)
    F_x_vector = np.zeros(6)
    F_x_vector[0] = F_x
    u_x = np.linalg.solve(K_condensed, F_x_vector)
    deflection_x_computed = u_x[0]
    assert np.isclose(deflection_x_computed, deflection_x_analytical, rtol=1e-06), f'X-direction deflection mismatch: computed={deflection_x_computed}, analytical={deflection_x_analytical}'