def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks:
    1) Analytical correctness against (EA/L)*[[1, -1], [-1, 1]]
    2) Shape and symmetry (2x2 and symmetric)
    3) Singularity (rigid body mode -> near-zero residual for [1, 1], and near-zero determinant)
    4) Integration consistency across 1-, 2-, and 3-point Gauss quadrature
    """
    import numpy as np
    E = 210000000000.0
    A = 0.003
    x_elem = np.array([0.25, 2.75], dtype=float)
    L = x_elem[1] - x_elem[0]
    assert L > 0.0
    scale = E * A / L
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    K_ref = scale * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    assert K2.shape == (2, 2)
    assert np.allclose(K2, K_ref, rtol=1e-12, atol=1e-12 * scale)
    assert K1.shape == (2, 2)
    assert K3.shape == (2, 2)
    assert np.allclose(K2, K2.T, rtol=1e-14, atol=1e-14 * scale)
    rigid = np.ones(2)
    residual = K2 @ rigid
    assert np.linalg.norm(residual, ord=2) <= 1e-12 * scale
    det2 = K2[0, 0] * K2[1, 1] - K2[0, 1] * K2[1, 0]
    assert abs(det2) <= 1e-10 * scale * scale
    assert np.allclose(K1, K2, rtol=1e-12, atol=1e-12 * scale)
    assert np.allclose(K3, K2, rtol=1e-12, atol=1e-12 * scale)