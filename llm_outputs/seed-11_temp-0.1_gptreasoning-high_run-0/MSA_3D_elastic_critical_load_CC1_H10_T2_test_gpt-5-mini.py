def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline with 10 elements and compare λ·P_ref
    to the analytical Euler cantilever value. Tolerance set for anticipated discretization error at 1e-5.
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 10
    P_ref = 1000.0
    radii = (0.5, 0.75, 1.0)
    lengths = (10.0, 20.0, 40.0)
    tol = 1e-05
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        for L in lengths:
            n_nodes = n_elem + 1
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([1.0, 0.0, 0.0])})
            boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
            nodal_loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam is not None and np.isfinite(lam) and (lam > 0)
            P_cr_analytic = np.pi ** 2 * E * I / (4.0 * L ** 2)
            expected_lambda = P_cr_analytic / P_ref
            rel_err = abs(lam - expected_lambda) / (abs(expected_lambda) + 1e-20)
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve cantilever in base orientation and after applying rigid-body rotation R
    to geometry, element axes and applied loads. Verify identical λ and that the
    rotated mode equals T @ mode_base up to scale and sign (T applies R to both translation and rotation DOFs).
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 10
    L = 10.0
    b = 0.05
    h = 0.15
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = b * h * (b ** 2 + h ** 2) / 12.0
    P_ref = 1000.0
    n_nodes = n_elem + 1
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
    local_z_base = np.array([1.0, 0.0, 0.0])
    elements_base = []
    for i in range(n_elem):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base.copy()})
    boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
    nodal_loads_base = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
    (lam_base, mode_base) = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)
    assert lam_base is not None and np.isfinite(lam_base) and (lam_base > 0)
    mode_base = np.asarray(mode_base, dtype=float)
    axis = np.array([0.3, 0.5, 0.8], dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(47.0)
    ux = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.cos(angle) * np.eye(3) + np.sin(angle) * ux + (1.0 - np.cos(angle)) * np.outer(axis, axis)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for el in elements_base:
        el_rot = el.copy()
        el_rot['local_z'] = (R @ np.asarray(el['local_z'], dtype=float)).copy()
        elements_rot.append(el_rot)
    nodal_loads_rot = {}
    for (ni, load) in nodal_loads_base.items():
        f = np.asarray(load[0:3], dtype=float)
        m = np.asarray(load[3:6], dtype=float)
        f_rot = R @ f
        m_rot = R @ m
        nodal_loads_rot[ni] = (f_rot[0], f_rot[1], f_rot[2], m_rot[0], m_rot[1], m_rot[2])
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot is not None and np.isfinite(lam_rot) and (lam_rot > 0)
    assert np.isclose(lam_base, lam_rot, rtol=1e-08, atol=1e-10)
    mode_rot = np.asarray(mode_rot, dtype=float)
    ndof = 6 * n_nodes
    T = np.zeros((ndof, ndof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    transformed_base = T @ mode_base
    denom = np.dot(transformed_base, transformed_base)
    assert denom > 0.0
    s = float(np.dot(transformed_base, mode_rot) / denom)
    diff_norm = np.linalg.norm(mode_rot - s * transformed_base)
    rel = diff_norm / (np.linalg.norm(mode_rot) + 1e-20)
    assert rel < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and the finest mesh achieves high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.75
    L = 10.0
    P_ref = 1000.0
    I = np.pi * r ** 4 / 4.0
    P_analytic = np.pi ** 2 * E * I / (4.0 * L ** 2)
    element_counts = (4, 8, 16, 32)
    errors = []
    for n_elem in element_counts:
        n_nodes = n_elem + 1
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
        A = np.pi * r ** 2
        J = np.pi * r ** 4 / 2.0
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([1.0, 0.0, 0.0])})
        boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
        nodal_loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam is not None and np.isfinite(lam) and (lam > 0)
        P_num = lam * P_ref
        rel_error = abs(P_num - P_analytic) / (abs(P_analytic) + 1e-20)
        errors.append(rel_error)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-08
    assert errors[-1] < 0.0001