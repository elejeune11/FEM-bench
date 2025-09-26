def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    shape check
    symmetry
    expected singularity due to rigid body modes
    block-level verification of axial, torsion, and bending terms"""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.33e-06
    Iz = 8.33e-06
    J = 1.67e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    eigenvalues = np.linalg.eigvalsh(K)
    zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-12)
    assert zero_eigenvalues == 6
    axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], axial_stiffness)
    assert np.isclose(K[0, 6], -axial_stiffness)
    assert np.isclose(K[6, 0], -axial_stiffness)
    assert np.isclose(K[6, 6], axial_stiffness)
    torsional_stiffness = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], torsional_stiffness)
    assert np.isclose(K[3, 9], -torsional_stiffness)
    assert np.isclose(K[9, 3], -torsional_stiffness)
    assert np.isclose(K[9, 9], torsional_stiffness)
    bending_stiffness_y = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], bending_stiffness_y)
    assert np.isclose(K[1, 5], 6 * E * Iz / L ** 2)
    assert np.isclose(K[1, 7], -bending_stiffness_y)
    assert np.isclose(K[1, 11], 6 * E * Iz / L ** 2)
    bending_stiffness_z = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], bending_stiffness_z)
    assert np.isclose(K[2, 4], -6 * E * Iy / L ** 2)
    assert np.isclose(K[2, 8], -bending_stiffness_z)
    assert np.isclose(K[2, 10], -6 * E * Iy / L ** 2)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory."""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 8.33e-06
    Iz = 8.33e-06
    J = 1.67e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    fixed_dofs = [0, 1, 2, 3, 4, 5]
    free_dofs = [6, 7, 8, 9, 10, 11]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_ff_inv = np.linalg.inv(K_ff)
    F_z = 1000.0
    F_y = 1000.0
    F_x = 1000.0
    F_free_z = np.array([0, 0, F_z, 0, 0, 0])
    u_free_z = K_ff_inv @ F_free_z
    w_tip_computed = u_free_z[2]
    w_tip_analytical = F_z * L ** 3 / (3 * E * Iy)
    assert np.isclose(w_tip_computed, w_tip_analytical, rtol=1e-10)
    F_free_y = np.array([0, F_y, 0, 0, 0, 0])
    u_free_y = K_ff_inv @ F_free_y
    v_tip_computed = u_free_y[1]
    v_tip_analytical = F_y * L ** 3 / (3 * E * Iz)
    assert np.isclose(v_tip_computed, v_tip_analytical, rtol=1e-10)
    F_free_x = np.array([F_x, 0, 0, 0, 0, 0])
    u_free_x = K_ff_inv @ F_free_x
    u_tip_computed = u_free_x[0]
    u_tip_analytical = F_x * L / (E * A)
    assert np.isclose(u_tip_computed, u_tip_analytical, rtol=1e-10)