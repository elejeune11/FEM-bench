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
    assert K.shape == (12, 12), 'Stiffness matrix should be 12x12'
    assert np.allclose(K, K.T, atol=1e-10), 'Stiffness matrix should be symmetric'
    eigenvalues = np.linalg.eigvalsh(K)
    num_zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-06)
    assert num_zero_eigenvalues == 6, 'Should have 6 zero eigenvalues for rigid body modes'
    k_axial = K[0, 0]
    expected_k_axial = E * A / L
    assert np.isclose(k_axial, expected_k_axial, rtol=1e-10), 'Axial stiffness term incorrect'
    k_torsion = K[3, 3]
    expected_k_torsion = E * J / (L * (1 + nu))
    assert np.isclose(k_torsion, expected_k_torsion, rtol=1e-10), 'Torsional stiffness term incorrect'
    k_bending_y = K[1, 1]
    expected_k_bending_y = 12 * E * Iz / L ** 3
    assert np.isclose(k_bending_y, expected_k_bending_y, rtol=1e-10), 'Bending stiffness in y direction incorrect'
    k_bending_z = K[2, 2]
    expected_k_bending_z = 12 * E * Iy / L ** 3
    assert np.isclose(k_bending_z, expected_k_bending_z, rtol=1e-10), 'Bending stiffness in z direction incorrect'
    assert K[0, 6] == -k_axial, 'Axial coupling between nodes should be negative'
    assert K[1, 7] == -k_bending_y, 'Bending coupling in y between nodes should be negative'
    assert K[2, 8] == -k_bending_z, 'Bending coupling in z between nodes should be negative'

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
    K_fixed = K[:6, :6]
    K_fixed_inv = np.linalg.inv(K_fixed)
    F_z = 1000.0
    f_z = np.array([0, 0, F_z, 0, 0, 0])
    u_z = K_fixed_inv @ f_z
    w_analytical_z = F_z * L ** 3 / (3 * E * Iy)
    assert np.isclose(u_z[2], w_analytical_z, rtol=1e-06), f'Z-direction deflection {u_z[2]} does not match analytical {w_analytical_z}'
    F_y = 1000.0
    f_y = np.array([0, F_y, 0, 0, 0, 0])
    u_y = K_fixed_inv @ f_y
    v_analytical_y = F_y * L ** 3 / (3 * E * Iz)
    assert np.isclose(u_y[1], v_analytical_y, rtol=1e-06), f'Y-direction deflection {u_y[1]} does not match analytical {v_analytical_y}'
    F_x = 1000.0
    f_x = np.array([F_x, 0, 0, 0, 0, 0])
    u_x = K_fixed_inv @ f_x
    u_analytical_x = F_x * L / (E * A)
    assert np.isclose(u_x[0], u_analytical_x, rtol=1e-06), f'X-direction deflection {u_x[0]} does not match analytical {u_analytical_x}'