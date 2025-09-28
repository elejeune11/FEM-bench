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
    x_elem = np.array([0.0, 2.0])
    E = 100.0
    A = 0.5
    L = x_elem[1] - x_elem[0]
    expected_factor = E * A / L
    K_2pt = fcn(x_elem, E, A, 2)
    expected_matrix = expected_factor * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_2pt, expected_matrix, rtol=1e-10, atol=1e-12)
    assert K_2pt.shape == (2, 2)
    assert np.allclose(K_2pt, K_2pt.T, rtol=1e-10, atol=1e-12)
    det = np.linalg.det(K_2pt)
    assert abs(det) < 1e-10
    K_1pt = fcn(x_elem, E, A, 1)
    K_3pt = fcn(x_elem, E, A, 3)
    assert np.allclose(K_1pt, K_2pt, rtol=1e-10, atol=1e-12)
    assert np.allclose(K_2pt, K_3pt, rtol=1e-10, atol=1e-12)
    assert np.allclose(K_1pt, K_3pt, rtol=1e-10, atol=1e-12)