def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    Cases:
    """
    tol = 1e-12
    Gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None)
    R_x = [[Gamma_x[i, j] for j in range(3)] for i in range(3)]
    R_x_expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for i in range(3):
        for j in range(3):
            assert abs(R_x[i][j] - R_x_expected[i][j]) <= tol
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, None)
    R_y = [[Gamma_y[i, j] for j in range(3)] for i in range(3)]
    R_y_expected = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    for i in range(3):
        for j in range(3):
            assert abs(R_y[i][j] - R_y_expected[i][j]) <= tol
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    R_z = [[Gamma_z[i, j] for j in range(3)] for i in range(3)]
    R_z_expected = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    for i in range(3):
        for j in range(3):
            assert abs(R_z[i][j] - R_z_expected[i][j]) <= tol

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Verifies orthonormality, right-handedness, block-diagonal structure with repeated 3x3 submatrices,
    and transformation invariants for sample beams, including a non-axial and a vertical beam."""
    tol = 1e-12

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    def cross(u, v):
        return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

    def norm(u):
        return (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5
    (x1, y1, z1) = (1.0, -2.0, 3.0)
    (x2, y2, z2) = (4.0, 0.0, 7.0)
    Gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    assert Gamma.shape == (12, 12)
    R = [[Gamma[i, j] for j in range(3)] for i in range(3)]
    for i in range(3):
        for k in range(3):
            s = R[i][0] * R[k][0] + R[i][1] * R[k][1] + R[i][2] * R[k][2]
            if i == k:
                assert abs(s - 1.0) <= tol
            else:
                assert abs(s) <= tol
    r1 = R[0]
    r2 = R[1]
    r3 = R[2]
    detR = dot(r1, cross(r2, r3))
    assert abs(detR - 1.0) <= 1e-10
    (dx, dy, dz) = (x2 - x1, y2 - y1, z2 - z1)
    L = (dx * dx + dy * dy + dz * dz) ** 0.5
    ex_g = [dx / L, dy / L, dz / L]
    local_from_axis = [R[0][0] * ex_g[0] + R[0][1] * ex_g[1] + R[0][2] * ex_g[2], R[1][0] * ex_g[0] + R[1][1] * ex_g[1] + R[1][2] * ex_g[2], R[2][0] * ex_g[0] + R[2][1] * ex_g[1] + R[2][2] * ex_g[2]]
    assert abs(local_from_axis[0] - 1.0) <= 1e-12
    assert abs(local_from_axis[1]) <= 1e-12
    assert abs(local_from_axis[2]) <= 1e-12
    for blk in range(4):
        i0 = 3 * blk
        for i in range(3):
            for j in range(3):
                assert abs(Gamma[i0 + i, i0 + j] - R[i][j]) <= 1e-12
        for other in range(4):
            if other == blk:
                continue
            j0 = 3 * other
            for i in range(3):
                for j in range(3):
                    assert abs(Gamma[i0 + i, j0 + j]) <= 1e-14
    GtG = Gamma.T @ Gamma
    for i in range(12):
        for j in range(12):
            expected = 1.0 if i == j else 0.0
            assert abs(GtG[i, j] - expected) <= 1e-10
    Gamma_vert = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 5.0, None)
    Rv = [[Gamma_vert[i, j] for j in range(3)] for i in range(3)]
    Rv_expected = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    for i in range(3):
        for j in range(3):
            assert abs(Rv[i][j] - Rv_expected[i][j]) <= tol
    for i in range(3):
        for k in range(3):
            s = Rv[i][0] * Rv[k][0] + Rv[i][1] * Rv[k][1] + Rv[i][2] * Rv[k][2]
            if i == k:
                assert abs(s - 1.0) <= tol
            else:
                assert abs(s) <= tol
    detRv = Rv[0][0] * (Rv[1][1] * Rv[2][2] - Rv[1][2] * Rv[2][1]) - Rv[0][1] * (Rv[1][0] * Rv[2][2] - Rv[1][2] * Rv[2][0]) + Rv[0][2] * (Rv[1][0] * Rv[2][1] - Rv[1][1] * Rv[2][0])
    assert abs(detRv - 1.0) <= 1e-12

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid inputs:
    1) Non-unit reference vector
    2) Reference vector parallel to beam axis
    3) Zero-length beam"""
    caught = False
    try:
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, [0.0, 0.0, 2.0])
    except ValueError:
        caught = True
    assert caught
    caught = False
    try:
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    except ValueError:
        caught = True
    assert caught
    caught = False
    try:
        fcn(1.0, -1.0, 2.0, 1.0, -1.0, 2.0, None)
    except ValueError:
        caught = True
    assert caught