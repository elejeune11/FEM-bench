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
    x_elem = np.array([0.0, 2.0])
    E = 100.0
    A = 0.5
    L = x_elem[1] - x_elem[0]
    EA_L = E * A / L
    expected_K = EA_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_1pt = fcn(x_elem, E, A, n_gauss=1)
    K_2pt = fcn(x_elem, E, A, n_gauss=2)
    K_3pt = fcn(x_elem, E, A, n_gauss=3)
    assert K_1pt.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert K_2pt.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert K_3pt.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(K_1pt, K_1pt.T), 'Stiffness matrix must be symmetric'
    assert np.allclose(K_2pt, K_2pt.T), 'Stiffness matrix must be symmetric'
    assert np.allclose(K_3pt, K_3pt.T), 'Stiffness matrix must be symmetric'
    assert np.allclose(K_2pt, expected_K), 'Stiffness matrix does not match analytical solution'
    assert np.allclose(K_1pt, K_2pt), '1-point and 2-point Gauss integration should yield same result'
    assert np.allclose(K_2pt, K_3pt), '2-point and 3-point Gauss integration should yield same result'
    det_K = np.linalg.det(K_2pt)
    assert np.isclose(det_K, 0.0), 'Stiffness matrix should be singular (zero determinant)'