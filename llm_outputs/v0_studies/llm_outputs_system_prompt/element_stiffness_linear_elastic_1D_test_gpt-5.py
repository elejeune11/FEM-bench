def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks multiple properties:
    1) Analytical correctness against (EA/L)*[[1,-1],[-1,1]]
    2) Shape and symmetry of the 2x2 stiffness matrix
    3) Singularity (zero determinant) for an unconstrained single element
    4) Integration consistency across 1-, 2-, and 3-point Gauss quadrature
    """
    E = 210000000000.0
    A = 0.003
    L = 2.5
    x_elem = [0.0, L]
    k = E * A / L
    expected = [[k, -k], [-k, k]]
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert hasattr(K1, 'shape') and K1.shape == (2, 2)
    assert hasattr(K2, 'shape') and K2.shape == (2, 2)
    assert hasattr(K3, 'shape') and K3.shape == (2, 2)
    tol = 1e-12 * max(1.0, abs(k))
    for i in range(2):
        for j in range(2):
            assert abs(K2[i][j] - expected[i][j]) <= tol
            assert abs(K1[i][j] - expected[i][j]) <= tol
            assert abs(K3[i][j] - expected[i][j]) <= tol
            assert abs(K1[i][j] - K2[i][j]) <= tol
            assert abs(K3[i][j] - K2[i][j]) <= tol
    assert abs(K2[0][1] - K2[1][0]) <= tol
    max_abs = max(abs(K2[0][0]), abs(K2[0][1]), abs(K2[1][0]), abs(K2[1][1]))
    det = K2[0][0] * K2[1][1] - K2[0][1] * K2[1][0]
    det_tol = 1e-10 * (max_abs * max_abs if max_abs > 0 else 1.0)
    assert abs(det) <= det_tol