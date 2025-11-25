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
    tol = 1e-12
    expected_K = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_2pt = fcn(x_elem, E, A, 2)
    assert np.allclose(K_2pt, expected_K, rtol=tol, atol=tol), 'Stiffness matrix does not match analytical solution'
    assert K_2pt.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(K_2pt, K_2pt.T, rtol=tol, atol=tol), 'Stiffness matrix must be symmetric'
    det_K = np.linalg.det(K_2pt)
    assert abs(det_K) < tol, f'Stiffness matrix should be singular (det={det_K}), but determinant is not zero'
    K_1pt = fcn(x_elem, E, A, 1)
    K_3pt = fcn(x_elem, E, A, 3)
    assert np.allclose(K_1pt, K_2pt, rtol=tol, atol=tol), '1-point Gauss quadrature gives different result from 2-point'
    assert np.allclose(K_3pt, K_2pt, rtol=tol, atol=tol), '3-point Gauss quadrature gives different result from 2-point'