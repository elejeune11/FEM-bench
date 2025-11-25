def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
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
    E = 200000000000.0
    A = 0.01
    L = 2.0
    x_elem = np.array([0.0, L])
    K_expected = E * A / L * np.array([[1, -1], [-1, 1]])
    K_actual = fcn(x_elem, E, A, n_gauss=2)
    assert np.allclose(K_actual, K_expected, rtol=1e-10), 'Stiffness matrix does not match analytical solution'
    assert K_actual.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(K_actual, K_actual.T, rtol=1e-10), 'Stiffness matrix must be symmetric'
    det = np.linalg.det(K_actual)
    assert abs(det) < 1e-10, f'Stiffness matrix should be singular (det={det}), but determinant is not near zero'
    K_gauss1 = fcn(x_elem, E, A, n_gauss=1)
    K_gauss2 = fcn(x_elem, E, A, n_gauss=2)
    K_gauss3 = fcn(x_elem, E, A, n_gauss=3)
    assert np.allclose(K_gauss1, K_gauss2, rtol=1e-12), '1-point and 2-point Gauss quadrature should yield identical results'
    assert np.allclose(K_gauss2, K_gauss3, rtol=1e-12), '2-point and 3-point Gauss quadrature should yield identical results'
    assert np.allclose(K_gauss1, K_gauss3, rtol=1e-12), '1-point and 3-point Gauss quadrature should yield identical results'