def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    Note: Minor floating-point differences may arise due to roundoff when summing weighted values.
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic.
    """
    E = 210000000000.0
    A = 0.0015
    x_elem = np.array([0.5, 2.0])
    L = x_elem[1] - x_elem[0]
    k_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_computed = fcn(x_elem, E, A, n_gauss=2)
    assert np.allclose(k_computed, k_analytical), 'Computed matrix does not match analytical solution.'
    assert k_computed.shape == (2, 2), f'Expected shape (2, 2), but got {k_computed.shape}.'
    assert np.allclose(k_computed, k_computed.T), 'Stiffness matrix is not symmetric.'
    assert np.isclose(np.linalg.det(k_computed), 0.0), 'Stiffness matrix should be singular (zero determinant).'
    k_1_point = fcn(x_elem, E, A, n_gauss=1)
    k_3_point = fcn(x_elem, E, A, n_gauss=3)
    assert np.allclose(k_1_point, k_analytical), '1-point Gauss quadrature result is incorrect.'
    assert np.allclose(k_computed, k_1_point), '2-point Gauss result differs from 1-point.'
    assert np.allclose(k_computed, k_3_point), '2-point Gauss result differs from 3-point.'
    x_elem_2 = np.array([-10.0, -8.0])
    L_2 = x_elem_2[1] - x_elem_2[0]
    k_analytical_2 = E * A / L_2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_computed_2 = fcn(x_elem_2, E, A, n_gauss=2)
    assert np.allclose(k_computed_2, k_analytical_2), 'Failed for element with negative coordinates.'