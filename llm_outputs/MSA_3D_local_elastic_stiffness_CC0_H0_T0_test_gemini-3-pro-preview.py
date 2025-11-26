def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.05
    L = 4.0
    Iy = 3e-05
    Iz = 6e-05
    J = 8e-05
    G = E / (2 * (1 + nu))
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12), 'Stiffness matrix must be 12x12'
    assert np.allclose(K, K.T, rtol=1e-08, atol=1e-08), 'Stiffness matrix must be symmetric'
    rank = np.linalg.matrix_rank(K)
    assert rank == 6, f'Rank of stiffness matrix should be 6, found {rank}'
    row_sums = np.sum(K, axis=1)
    assert np.allclose(row_sums, 0, atol=1e-05), 'Rows should sum to zero (translational equilibrium)'
    k_axial = E * A / L
    assert np.isclose(K[0, 0], k_axial), 'Axial stiffness u1 mismatch'
    assert np.isclose(K[0, 6], -k_axial), 'Axial coupling u1-u2 mismatch'
    assert np.isclose(K[6, 6], k_axial), 'Axial stiffness u2 mismatch'
    k_torsion = G * J / L
    assert np.isclose(K[3, 3], k_torsion), 'Torsional stiffness thx1 mismatch'
    assert np.isclose(K[3, 9], -k_torsion), 'Torsional coupling mismatch'
    assert np.isclose(K[9, 9], k_torsion), 'Torsional stiffness thx2 mismatch'
    k_bend_z = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], k_bend_z), 'Bending stiffness v1 (Iy-based) mismatch'
    k_bend_y = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], k_bend_y), 'Bending stiffness w1 (Iz-based) mismatch'

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    E = 200000000000.0
    nu = 0.25
    A = 0.01
    L = 5.0
    Iy = 2e-05
    Iz = 4e-05
    J = 0.0001
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_reduced = K[6:, 6:]
    P_z = 1000.0
    F_z = np.zeros(6)
    F_z[2] = P_z
    u_z = np.linalg.solve(K_reduced, F_z)
    w_calc = u_z[2]
    w_analytical = P_z * L ** 3 / (3 * E * Iy)
    assert np.isclose(w_calc, w_analytical, rtol=1e-05), f'Z-deflection mismatch. Calc: {w_calc}, Analytical: {w_analytical}'
    P_y = 1000.0
    F_y = np.zeros(6)
    F_y[1] = P_y
    u_y = np.linalg.solve(K_reduced, F_y)
    v_calc = u_y[1]
    v_analytical = P_y * L ** 3 / (3 * E * Iz)
    assert np.isclose(v_calc, v_analytical, rtol=1e-05), f'Y-deflection mismatch. Calc: {v_calc}, Analytical: {v_analytical}'
    P_x = 5000.0
    F_x = np.zeros(6)
    F_x[0] = P_x
    u_x = np.linalg.solve(K_reduced, F_x)
    u_calc = u_x[0]
    u_analytical = P_x * L / (E * A)
    assert np.isclose(u_calc, u_analytical, rtol=1e-05), f'Axial deflection mismatch. Calc: {u_calc}, Analytical: {u_analytical}'