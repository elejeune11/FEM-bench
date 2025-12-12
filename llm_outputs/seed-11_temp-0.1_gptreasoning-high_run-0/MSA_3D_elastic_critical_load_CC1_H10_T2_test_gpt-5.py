def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements and verify relative error is small (accounting for discretization).
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    ne = 10
    tol_rel = 0.003
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in lengths:
            dz = L / ne
            node_coords = np.array([[0.0, 0.0, i * dz] for i in range(ne + 1)], dtype=float)
            elements = []
            for i in range(ne):
                elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I, I_z=I, J=J, local_z=np.array([0.0, 1.0, 0.0], dtype=float)))
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {ne: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            assert isinstance(mode, np.ndarray) and mode.shape == (6 * (ne + 1),)
            p_euler = np.pi ** 2 / 4.0 * E * I / L ** 2
            rel_err = abs(lam - p_euler) / p_euler
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
    mode_rot ≈ T @ mode_base, allowing for arbitrary scale and sign.
    """
    E = 210000000000.0
    nu = 0.3
    L = 15.0
    ne = 12
    b_z = 0.3
    h_y = 0.2
    A = b_z * h_y
    I_y = b_z * h_y ** 3 / 12.0
    I_z = h_y * b_z ** 3 / 12.0
    J = I_y + I_z
    dz = L / ne
    node_coords = np.array([[0.0, 0.0, i * dz] for i in range(ne + 1)], dtype=float)
    base_local_z = np.array([0.0, 1.0, 0.0], dtype=float)
    elements = [dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=base_local_z) for i in range(ne)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {ne: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert lam_base > 0.0
    assert isinstance(mode_base, np.ndarray) and mode_base.shape == (6 * (ne + 1),)
    (ax, ay, az) = (0.3, -0.5, 0.1)
    (cx, sx) = (np.cos(ax), np.sin(ax))
    (cy, sy) = (np.cos(ay), np.sin(ay))
    (cz, sz) = (np.cos(az), np.sin(az))
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    R = R_z @ R_y @ R_x
    node_coords_rot = (node_coords @ R.T).astype(float)
    local_z_rot = (R @ base_local_z).astype(float)
    elements_rot = [dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z_rot) for i in range(ne)]
    F_ref = np.array([0.0, 0.0, -1.0], dtype=float)
    F_rot = (R @ F_ref).tolist()
    nodal_loads_rot = {ne: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    assert isinstance(mode_rot, np.ndarray) and mode_rot.shape == (6 * (ne + 1),)
    assert abs(lam_rot - lam_base) / lam_base < 1e-06
    N = ne + 1
    T = np.zeros((6 * N, 6 * N), dtype=float)
    for i in range(N):
        i6 = 6 * i
        T[i6:i6 + 3, i6:i6 + 3] = R
        T[i6 + 3:i6 + 6, i6 + 3:i6 + 6] = R
    b = T @ mode_base
    a = mode_rot
    nb = np.dot(b, b)
    if nb > 0:
        s = float(np.dot(a, b) / nb)
    else:
        s = 0.0
    denom = max(np.linalg.norm(a), 1e-14)
    rel_res = np.linalg.norm(a - s * b) / denom
    assert rel_res < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The numerical critical load should approach the analytical Euler value as the mesh is refined,
    and the finest mesh should achieve very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.5
    L = 20.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    ne_list = [2, 4, 8, 16, 32]
    p_euler = np.pi ** 2 / 4.0 * E * I / L ** 2
    errors = []
    for ne in ne_list:
        dz = L / ne
        node_coords = np.array([[0.0, 0.0, i * dz] for i in range(ne + 1)], dtype=float)
        elements = []
        for i in range(ne):
            elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I, I_z=I, J=J, local_z=np.array([0.0, 1.0, 0.0], dtype=float)))
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {ne: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        assert isinstance(mode, np.ndarray) and mode.shape == (6 * (ne + 1),)
        errors.append(abs(lam - p_euler) / p_euler)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-12
    assert errors[-1] < 0.001